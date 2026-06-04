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


def _pixel_to_spherical(cx, cy, frame_w, frame_h):
    """Equirectangularフレーム上のピクセル座標を球面座標に変換する。

    Parameters
    ----------
    cx, cy : float
        ピクセル中心座標。
    frame_w, frame_h : int
        フレームの幅・高さ。

    Returns
    -------
    yaw_deg : float   方位角 [-180, 180]
    pitch_deg : float 仰角   [-90,   90]
    """
    yaw_deg = (cx / frame_w - 0.5) * 360.0
    pitch_deg = (0.5 - cy / frame_h) * 180.0
    return float(yaw_deg), float(pitch_deg)


class PersonDetector:
    """YOLOv8 による人物検出ラッパー。

    Equirectangular フレームから人物を検出し、
    各人物の球面座標（方位角・仰角）を返す。

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

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            yaw, pitch = _pixel_to_spherical(cx, cy, w, h)
            detections.append(Detection(
                yaw_deg=yaw,
                pitch_deg=pitch,
                bbox_px=(x1, y1, x2, y2),
                confidence=float(box.conf[0]),
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
