from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

_YOLO_PERSON_CLASS = 0


@dataclass
class Detection:
    """検出された1人分の情報。"""
    yaw_deg: float    # 方位角 [-180, 180]
    pitch_deg: float  # 仰角   [-90,   90]
    bbox_px: tuple    # Equirectangular上のBBox (x1, y1, x2, y2) ピクセル座標
    confidence: float
    extra_bbox_px: tuple = None  # 360度境界マージ時の2片目（描画用）。通常は None


def _pixel_to_spherical(cx, cy, frame_w, frame_h):
    """Equirectangularフレーム上のピクセル座標を球面座標に変換する。

    Parameters
    ----------
    cx, cy : float
        ピクセル中心座標。cx は境界マージ時に負になり得る（左へ w シフトした場合）。
    frame_w, frame_h : int
        フレームの幅・高さ。

    Returns
    -------
    yaw_deg : float   方位角 [-180, 180]
    pitch_deg : float 仰角   [-90,   90]
    """
    yaw_deg = (cx / frame_w - 0.5) * 360.0
    # cx が範囲外（境界マージで負など）の場合のみ [-180, 180] に補正
    if yaw_deg > 180.0:
        yaw_deg -= 360.0
    elif yaw_deg < -180.0:
        yaw_deg += 360.0
    pitch_deg = (0.5 - cy / frame_h) * 180.0
    return float(yaw_deg), float(pitch_deg)


def _merge_wraparound(boxes, w, h, edge_ratio=0.02):
    """Equirectangular の左右端にまたがる同一人物の BBox を統合する。

    左端接触（x1 < w*edge_ratio）の片と右端接触（x2 > w*(1-edge_ratio)）の片で
    y 方向が重なるものを、つなぎ目をまたぐ1人として統合する。

    Parameters
    ----------
    boxes : list[tuple]
        (x1, y1, x2, y2, conf) のリスト。
    w, h : int
        フレームの幅・高さ。
    edge_ratio : float
        端とみなす幅の割合。

    Returns
    -------
    list[tuple]
        (cx_for_yaw, cy, primary_bbox, conf, extra_bbox) のリスト。
        cx_for_yaw は yaw 計算用の中心 x（マージ時は負になり得る）。
        extra_bbox はマージ時の2片目（なければ None）。
    """
    edge = w * edge_ratio
    used = [False] * len(boxes)
    result = []

    for i, bi in enumerate(boxes):
        if used[i]:
            continue
        x1i, y1i, x2i, y2i, ci = bi
        merged = False

        for j, bj in enumerate(boxes):
            if i == j or used[j]:
                continue
            x1j, y1j, x2j, y2j, cj = bj

            # 一方が左端接触・他方が右端接触か判定
            left = right = None
            if x1i < edge and x2j > w - edge:
                left, right = bi, bj
            elif x1j < edge and x2i > w - edge:
                left, right = bj, bi
            if left is None:
                continue

            # y 方向の重なり判定
            ly1, ly2 = left[1], left[3]
            ry1, ry2 = right[1], right[3]
            if min(ly2, ry2) <= max(ly1, ry1):
                continue  # 重ならない → 別人物

            # 右片を左へ w シフトして連結した中心 x（負になり得る）
            cx_for_yaw = ((right[0] - w) + left[2]) / 2.0
            cy = (left[1] + left[3] + right[1] + right[3]) / 4.0
            conf = max(left[4], right[4])
            result.append((cx_for_yaw, cy, left[:4], conf, right[:4]))
            used[i] = used[j] = True
            merged = True
            break

        if not merged and not used[i]:
            cx = (x1i + x2i) / 2.0
            cy = (y1i + y2i) / 2.0
            result.append((cx, cy, (x1i, y1i, x2i, y2i), ci, None))
            used[i] = True

    return result


class PersonDetector:
    """YOLOv8 による人物検出ラッパー。

    Equirectangular フレームから人物を検出し、
    各人物の球面座標（方位角・仰角）を返す。
    左右端にまたがる人物は1人に統合する（360度境界マージ）。

    Usage:
        detector = PersonDetector()
        detections = detector.detect(frame_bgr)
        for d in detections:
            print(d.yaw_deg, d.pitch_deg, d.confidence)
    """

    def __init__(self, model_size="n", conf=0.5):
        """
        Parameters
        ----------
        model_size : str
            YOLOv8 のサイズ。"n" (nano) / "s" / "m" / "l" / "x"。
            デフォルトは最軽量の "n"。
        conf : float
            検出信頼スコアの閾値。これを下回る検出は除外する。
        """
        self.conf = conf
        self._model = YOLO(f"yolov8{model_size}.pt")
        print(f"[PersonDetector] YOLOv8{model_size} ロード完了")

    def detect(self, frame_bgr):
        """Equirectangularフレームから人物を検出する。

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR uint8 equirectangular フレーム shape (H, W, 3)。
            THETA X から取得したフレームをそのまま渡せる。

        Returns
        -------
        list[Detection]
            検出された人物リスト。confidence の高い順に並んでいる。
            人物が1人も検出されなかった場合は空リストを返す。
        """
        h, w = frame_bgr.shape[:2]

        results = self._model(
            frame_bgr,
            classes=[_YOLO_PERSON_CLASS],
            conf=self.conf,
            verbose=False,
        )

        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((x1, y1, x2, y2, float(box.conf[0])))

        # 360度境界をまたぐ人物を統合
        merged = _merge_wraparound(boxes, w, h)

        detections = []
        for cx_for_yaw, cy, primary_bbox, conf, extra_bbox in merged:
            yaw, pitch = _pixel_to_spherical(cx_for_yaw, cy, w, h)
            detections.append(Detection(
                yaw_deg=yaw,
                pitch_deg=pitch,
                bbox_px=primary_bbox,
                confidence=conf,
                extra_bbox_px=extra_bbox,
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
