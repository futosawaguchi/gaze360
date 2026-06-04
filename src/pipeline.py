"""
gaze360 メインパイプライン。

THETA X からの 360° 映像に対してリアルタイムで視線推定を行い、
Equirectangular フレーム上に結果を可視化する。

実行方法:
    python -m src.pipeline               # THETA X カメラ使用
    python -m src.pipeline --source path/to/video.mp4  # 動画ファイル使用

操作:
    q : 終了
    s : 現在のフレームを outputs/ に保存
"""

import argparse
import os
import time

import cv2
import numpy as np

from src.camera.theta_stream import ThetaStream
from src.detection.person_detector import PersonDetector
from src.gaze.estimator import GazeEstimator
from src.projection.equirect import equirect_to_perspective, heatmap_to_spherical

# ---- 定数 ----------------------------------------------------------------
FOV_DEG = 90          # 透視投影パッチの画角
PATCH_SIZE = 448      # Gaze-LLE の入力サイズ
INOUT_THRESH = 0.3    # これ未満は「フレーム外を見ている」と判定
DISPLAY_SCALE = 0.5   # 表示用にフレームを縮小する倍率（1.0 = 原寸）

# ---- 描画色 ---------------------------------------------------------------
COLOR_PERSON  = (0, 255, 0)    # 人物 BBox: 緑
COLOR_GAZE_IN = (0, 60, 255)   # 視線先（フレーム内）: 赤
COLOR_GAZE_OUT = (160, 160, 160) # 視線先（フレーム外）: グレー
COLOR_FPS     = (0, 220, 255)  # FPS テキスト: 黄


def _spherical_to_pixel(yaw_deg, pitch_deg, frame_w, frame_h):
    """球面座標 → Equirectangular フレームのピクセル座標。"""
    px = int((yaw_deg / 360.0 + 0.5) * frame_w)
    py = int((0.5 - pitch_deg / 180.0) * frame_h)
    px = max(0, min(frame_w - 1, px))
    py = max(0, min(frame_h - 1, py))
    return px, py


class GazePipeline:
    """全モジュールを統合したリアルタイム視線推定パイプライン。"""

    def __init__(self, source=None, display_scale=DISPLAY_SCALE):
        """
        Parameters
        ----------
        source : str | None
            None のとき THETA X カメラを使用。
            ファイルパスを渡すと動画ファイルから読み込む（デバッグ用）。
        display_scale : float
            表示ウィンドウの縮小倍率。
        """
        self.source = source
        self.display_scale = display_scale

        print("=" * 50)
        print("  gaze360 パイプライン 初期化")
        print("=" * 50)
        self.detector = PersonDetector(model_size="n")
        self.estimator = GazeEstimator()
        self._fps = 0.0
        os.makedirs("outputs", exist_ok=True)

    # ------------------------------------------------------------------
    # メインループ
    # ------------------------------------------------------------------

    def run(self):
        """パイプラインを起動してメインループを実行する。"""
        print("\n起動完了。'q' で終了、's' でフレーム保存\n")

        if self.source is None:
            self._run_camera()
        else:
            self._run_video(self.source)

    def _run_camera(self):
        with ThetaStream() as stream:
            while True:
                frame = stream.read()
                if frame is None:
                    print("フレーム取得失敗。ストリームが終了しました。")
                    break
                if not self._step(frame):
                    break
        cv2.destroyAllWindows()

    def _run_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"動画ファイルを開けません: {path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not self._step(frame):
                break
        cap.release()
        cv2.destroyAllWindows()

    def _step(self, frame):
        """1フレーム処理してウィンドウ更新。False を返したら終了。"""
        t0 = time.perf_counter()

        annotated, logs = self._process(frame)

        elapsed = time.perf_counter() - t0
        self._fps = 0.85 * self._fps + 0.15 * (1.0 / max(elapsed, 1e-6))

        for log in logs:
            print(log)

        display = self._resize_for_display(annotated)
        cv2.imshow("gaze360", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if key == ord("s"):
            ts = int(time.time())
            save_path = f"outputs/frame_{ts}.jpg"
            cv2.imwrite(save_path, annotated)
            print(f"[保存] {save_path}")
        return True

    # ------------------------------------------------------------------
    # 推論処理
    # ------------------------------------------------------------------

    def _process(self, frame):
        """1フレームに対して検出→推論→ログ生成を行う。"""
        h, w = frame.shape[:2]
        detections = self.detector.detect(frame)

        logs = []
        gaze_results = []

        for i, det in enumerate(detections):
            patch_rgb = equirect_to_perspective(
                frame,
                yaw_deg=det.yaw_deg,
                pitch_deg=det.pitch_deg,
                fov_deg=FOV_DEG,
                out_size=PATCH_SIZE,
            )
            heatmap, inout = self.estimator.predict(patch_rgb)
            gaze_yaw, gaze_pitch = heatmap_to_spherical(
                heatmap,
                yaw_deg=det.yaw_deg,
                pitch_deg=det.pitch_deg,
                fov_deg=FOV_DEG,
            )
            gaze_results.append((gaze_yaw, gaze_pitch, inout))

            status = "フレーム内" if inout >= INOUT_THRESH else "フレーム外"
            logs.append(
                f"人物{i + 1}: 方位角 {gaze_yaw:+.1f}°, 仰角 {gaze_pitch:+.1f}° を見ている"
                f"  [{status}  inout={inout:.2f}  conf={det.confidence:.2f}]"
            )

        if not detections:
            logs.append(f"[フレーム] 人物なし  FPS={self._fps:.1f}")

        annotated = self._draw(frame, detections, gaze_results)
        return annotated, logs

    # ------------------------------------------------------------------
    # 可視化
    # ------------------------------------------------------------------

    def _draw(self, frame, detections, gaze_results):
        vis = frame.copy()
        h, w = vis.shape[:2]

        for i, (det, (gaze_yaw, gaze_pitch, inout)) in enumerate(
            zip(detections, gaze_results)
        ):
            x1, y1, x2, y2 = (int(v) for v in det.bbox_px)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 人物 BBox（緑）
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_PERSON, 2)
            cv2.putText(
                vis, f"P{i + 1}",
                (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_PERSON, 2,
            )

            # 視線先ピクセル
            gx, gy = _spherical_to_pixel(gaze_yaw, gaze_pitch, w, h)
            color = COLOR_GAZE_IN if inout >= INOUT_THRESH else COLOR_GAZE_OUT

            # 矢印（yaw差が大きい場合は境界またぎのため省略）
            yaw_diff = abs(gaze_yaw - det.yaw_deg)
            if yaw_diff < 160:
                cv2.arrowedLine(vis, (cx, cy), (gx, gy), color, 2, tipLength=0.15)

            # 視線先マーカー（塗りつぶし円）
            cv2.circle(vis, (gx, gy), 10, color, -1)
            cv2.circle(vis, (gx, gy), 10, (255, 255, 255), 1)  # 白縁

            # 方位角・仰角テキスト
            label = f"az:{gaze_yaw:+.0f} el:{gaze_pitch:+.0f}"
            cv2.putText(
                vis, label,
                (gx + 13, gy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

        if not detections:
            cv2.putText(
                vis, "No persons detected",
                (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2,
            )

        # FPS（右上）
        cv2.putText(
            vis, f"FPS: {self._fps:.1f}",
            (w - 130, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_FPS, 2,
        )

        return vis

    def _resize_for_display(self, frame):
        if self.display_scale == 1.0:
            return frame
        h, w = frame.shape[:2]
        return cv2.resize(
            frame,
            (int(w * self.display_scale), int(h * self.display_scale)),
            interpolation=cv2.INTER_LINEAR,
        )


# ------------------------------------------------------------------
# エントリポイント
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="gaze360 リアルタイム視線推定パイプライン")
    parser.add_argument(
        "--source", default=None,
        help="動画ファイルパス（省略時は THETA X カメラを使用）",
    )
    parser.add_argument(
        "--scale", type=float, default=DISPLAY_SCALE,
        help=f"表示ウィンドウの縮小倍率（デフォルト: {DISPLAY_SCALE}）",
    )
    args = parser.parse_args()

    pipeline = GazePipeline(source=args.source, display_scale=args.scale)
    pipeline.run()


if __name__ == "__main__":
    main()
