import cv2
import numpy as np
from ultralytics import YOLO

# COCO pose keypoint インデックス（顔まわり）
_KP_NOSE = 0
_KP_LEFT_EYE = 1
_KP_RIGHT_EYE = 2
_KP_LEFT_EAR = 3
_KP_RIGHT_EAR = 4
_FACE_KPS = [_KP_NOSE, _KP_LEFT_EYE, _KP_RIGHT_EYE, _KP_LEFT_EAR, _KP_RIGHT_EAR]

_KP_CONF_THRESH = 0.3   # この信頼度未満の keypoint は無視


class HeadDetector:
    """YOLOv8-pose による頭部 BBox 検出ラッパー。

    透視投影パッチ内で人物の姿勢推定を行い、顔まわりの keypoint から
    頭部バウンディングボックスを構成して Gaze-LLE に渡す用途。

    Usage:
        detector = HeadDetector()
        bbox = detector.detect_head(patch_rgb)  # (xmin,ymin,xmax,ymax) 正規化 or None
    """

    def __init__(self, model_size="n", conf=0.3):
        """
        Parameters
        ----------
        model_size : str
            YOLOv8-pose のサイズ。"n"/"s"/"m"/"l"/"x"。デフォルトは最軽量 "n"。
        conf : float
            人物検出の信頼スコア閾値。
        """
        self.conf = conf
        self._model = YOLO(f"yolov8{model_size}-pose.pt")
        print(f"[HeadDetector] YOLOv8{model_size}-pose ロード完了")

    def detect_head(self, patch_rgb):
        """透視投影パッチ内の頭部 BBox を正規化座標で返す。

        Parameters
        ----------
        patch_rgb : np.ndarray
            RGB uint8 透視投影パッチ shape (H, W, 3)。
            equirect_to_perspective() の出力をそのまま渡せる。

        Returns
        -------
        tuple | None
            頭部 BBox (xmin, ymin, xmax, ymax) の正規化座標 [0, 1]。
            人物・顔が検出できなければ None。
        """
        h, w = patch_rgb.shape[:2]
        # ultralytics は BGR 入力を前提とするため変換
        patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)

        results = self._model(patch_bgr, conf=self.conf, verbose=False)
        result = results[0]

        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        # keypoints.data: (人数, 17, 3) = (x, y, conf)
        kps_all = result.keypoints.data.cpu().numpy()

        # パッチ中心に最も近い人物を採用（パッチは対象人物を中心に切り出している）
        best_kps = self._select_center_person(kps_all, w, h)
        if best_kps is None:
            return None

        return self._face_kps_to_head_bbox(best_kps, w, h)

    # ------------------------------------------------------------------
    # 内部処理
    # ------------------------------------------------------------------

    @staticmethod
    def _select_center_person(kps_all, w, h):
        """パッチ中心に最も近い人物の keypoint 配列を返す。"""
        cx_img, cy_img = w / 2.0, h / 2.0
        best = None
        best_dist = float("inf")
        for kps in kps_all:
            face = kps[_FACE_KPS]
            visible = face[face[:, 2] >= _KP_CONF_THRESH]
            if len(visible) == 0:
                continue
            # 顔 keypoint の重心とパッチ中心の距離
            fcx = visible[:, 0].mean()
            fcy = visible[:, 1].mean()
            dist = (fcx - cx_img) ** 2 + (fcy - cy_img) ** 2
            if dist < best_dist:
                best_dist = dist
                best = kps
        return best

    @staticmethod
    def _face_kps_to_head_bbox(kps, w, h):
        """顔まわり keypoint から頭部 BBox（正規化座標）を構成する。"""
        face = kps[_FACE_KPS]
        visible = face[face[:, 2] >= _KP_CONF_THRESH]
        if len(visible) == 0:
            return None

        xs = visible[:, 0]
        ys = visible[:, 1]
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())

        # 顔 keypoint の外接矩形に、頭全体を含むようマージンを付ける
        bw = max(xmax - xmin, 1.0)
        bh = max(ymax - ymin, 1.0)
        # 横は左右に40%ずつ、下は顎方向に60%、上は頭頂方向に高さの150%拡張
        xmin -= bw * 0.4
        xmax += bw * 0.4
        ymax += bh * 0.6
        ymin -= bh * 1.5

        # 画像範囲にクランプして正規化
        xmin = max(0.0, xmin) / w
        ymin = max(0.0, ymin) / h
        xmax = min(float(w), xmax) / w
        ymax = min(float(h), ymax) / h

        if xmax <= xmin or ymax <= ymin:
            return None
        return (xmin, ymin, xmax, ymax)
