"""
gaze360 メインパイプライン。

THETA X からの 360° 映像に対してリアルタイムで視線推定を行い、
Equirectangular フレーム上に結果を可視化する。

実行方法:
    python -m src.pipeline                          # ローカル: cv2 ウィンドウ表示
    python -m src.pipeline --source video.mp4       # 動画ファイル使用
    python -m src.pipeline --stream-port 8080       # GPU サーバー: ブラウザで視聴

操作（ローカルモード）:
    q : 終了
    s : 現在のフレームを outputs/ に保存

操作（ストリームモード）:
    Ctrl+C : 終了
    ブラウザで http://<サーバーIP>:<ポート> を開く
"""

import argparse
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from typing import Callable, List, Optional

import cv2
import numpy as np

from src.camera.theta_stream import ThetaStream
from src.detection.person_detector import PersonDetector
from src.detection.head_detector import HeadDetector
from src.gaze.estimator import GazeEstimator
from src.gaze.result import GazeResult
from src.projection.equirect import (
    equirect_to_perspective,
    heatmap_to_spherical,
    patch_norm_to_spherical,
)

# ---- 定数 ----------------------------------------------------------------
FOV_DEG = 90          # 透視投影パッチの画角
PATCH_SIZE = 448      # Gaze-LLE の入力サイズ
INOUT_THRESH = 0.3    # これ未満は「フレーム外を見ている」と判定
DISPLAY_SCALE = 1.0   # 表示用にフレームを縮小する倍率（1.0 = 原寸）
STREAM_SCALE = 0.5    # ストリーム配信時のデフォルト縮小倍率（VPN帯域対策）
STREAM_JPEG_QUALITY = 55  # ストリーム配信時の JPEG 品質（低いほど軽い）

# ---- 描画色 ---------------------------------------------------------------
COLOR_PERSON  = (0, 255, 0)    # 人物 BBox: 緑
COLOR_HEAD    = (255, 0, 255)  # 頭部 BBox: マゼンタ
COLOR_GAZE_IN = (0, 60, 255)   # 視線先（フレーム内）: 赤
COLOR_GAZE_OUT = (160, 160, 160) # 視線先（フレーム外）: グレー
COLOR_FPS     = (0, 220, 255)  # FPS テキスト: 黄


class _MJPEGHandler(BaseHTTPRequestHandler):
    """MJPEG ストリーミング用 HTTP ハンドラー。"""

    def do_GET(self):
        if self.path == "/stream":
            self._send_stream()
        else:
            self._send_index()

    def _send_index(self):
        html = (
            b"<html><head><title>gaze360</title></head>"
            b'<body style="background:#000;margin:0">'
            b'<img src="/stream" style="width:100%">'
            b"</body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html)

    def _send_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        mjpeg = self.server.mjpeg_server
        try:
            while True:
                with mjpeg.frame_lock:
                    frame = mjpeg.latest_frame
                if frame is not None:
                    ok, jpeg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, mjpeg.quality]
                    )
                    if ok:
                        data = jpeg.tobytes()
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + data
                            + b"\r\n"
                        )
                time.sleep(0.04)  # ~25FPS 上限
        except Exception:
            pass

    def log_message(self, *args):
        pass  # アクセスログを抑制


class MJPEGServer:
    """バックグラウンドスレッドで動く MJPEG HTTP ストリームサーバー。

    ブラウザで http://<IP>:<port> を開くと映像が見える。
    GPU サーバー上でディスプレイなしで動かすときに使用する。
    """

    def __init__(self, port: int, quality: int = STREAM_JPEG_QUALITY):
        self.port = port
        self.quality = quality  # JPEG 品質（高いほど高画質・高帯域）
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self._server = None

    def update(self, frame_bgr):
        """表示フレームを更新する（パイプラインのメインループから呼ぶ）。"""
        with self.frame_lock:
            self.latest_frame = frame_bgr.copy()

    def start(self):
        """HTTP サーバーをバックグラウンドスレッドで起動する。"""
        # allow_reuse_address=True により、前回の終了直後でも同じポートで再起動できる
        HTTPServer.allow_reuse_address = True
        server = HTTPServer(("0.0.0.0", self.port), _MJPEGHandler)
        server.mjpeg_server = self
        self._server = server
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[MJPEGServer] ブラウザで http://<サーバーIP>:{self.port} を開いてください")

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()  # ソケットを明示的に解放
            self._server = None


def _spherical_to_pixel(yaw_deg, pitch_deg, frame_w, frame_h):
    """球面座標 → Equirectangular フレームのピクセル座標。"""
    px = int((yaw_deg / 360.0 + 0.5) * frame_w)
    py = int((0.5 - pitch_deg / 180.0) * frame_h)
    px = max(0, min(frame_w - 1, px))
    py = max(0, min(frame_h - 1, py))
    return px, py


def _draw_gaze_arrow(vis, ox, oy, origin_yaw, gaze_yaw, gx, gy, color, w):
    """頭部起点 (ox, oy) → 視線先 (gx, gy) の矢印を描く。

    ±180°のつなぎ目をまたぐ視線（頭部と視線先が画面の左右逆端にある）は、
    最短角差で進む向きに沿って2分割で描画し、画面を横断する直線を防ぐ。
    """
    # 最短角差 [-180, 180]。これに沿った「巻き戻さない」視線先 x を求める。
    dyaw = ((gaze_yaw - origin_yaw + 180.0) % 360.0) - 180.0
    gx_unwrapped = ox + dyaw / 360.0 * w

    # つなぎ目をまたがない通常ケース
    if 0 <= gx_unwrapped <= w - 1:
        cv2.arrowedLine(vis, (ox, oy), (gx, gy), color, 2, tipLength=0.15)
        return

    # つなぎ目をまたぐ: 出口の端で分割し、反対の端から続ける
    if gx_unwrapped > w - 1:          # 右端から出て左端へ続く
        edge_out, edge_in = w - 1, 0
        t = (w - 1 - ox) / (gx_unwrapped - ox)
    else:                             # 左端から出て右端へ続く
        edge_out, edge_in = 0, w - 1
        t = ox / (ox - gx_unwrapped)
    y_seam = int(round(oy + t * (gy - oy)))
    cv2.line(vis, (ox, oy), (edge_out, y_seam), color, 2)                        # 出口（矢じりなし）
    cv2.arrowedLine(vis, (edge_in, y_seam), (gx, gy), color, 2, tipLength=0.15)  # 続き（矢じり）


def _draw_head_box(vis, corner_a, corner_b, color, w, h):
    """頭部 BBox を equirect フレーム上に矩形描画する。

    corner_a / corner_b は (azimuth_deg, elevation_deg) の対角2隅。つなぎ目をまたぐ
    場合（ピクセル幅が画面幅の半分超）は左右2枚の矩形に分割して描く。
    """
    ax, ay = _spherical_to_pixel(corner_a[0], corner_a[1], w, h)
    bx, by = _spherical_to_pixel(corner_b[0], corner_b[1], w, h)
    ytop, ybot = min(ay, by), max(ay, by)

    if abs(bx - ax) <= w / 2:        # 通常
        cv2.rectangle(vis, (min(ax, bx), ytop), (max(ax, bx), ybot), color, 2)
    else:                            # つなぎ目またぎ: 左端側と右端側に分割
        cv2.rectangle(vis, (0, ytop), (min(ax, bx), ybot), color, 2)
        cv2.rectangle(vis, (max(ax, bx), ytop), (w - 1, ybot), color, 2)


class GazePipeline:
    """全モジュールを統合したリアルタイム視線推定パイプライン。"""

    def __init__(self, source=None, display_scale=DISPLAY_SCALE, stream_port=None,
                 stream_quality=STREAM_JPEG_QUALITY,
                 on_results: Optional[Callable[[List[GazeResult]], None]] = None):
        """
        Parameters
        ----------
        source : str | None
            None のとき THETA X カメラを使用。
            ファイルパスを渡すと動画ファイルから読み込む（デバッグ用）。
        display_scale : float
            表示ウィンドウの縮小倍率。
        stream_port : int | None
            指定時は MJPEG HTTP サーバーを起動してブラウザで映像を視聴できる。
            None のとき cv2.imshow でローカル表示（デフォルト）。
        stream_quality : int
            ストリーム配信時の JPEG 品質（高いほど高画質・高帯域）。
        on_results : Callable[[list[GazeResult]], None] | None
            各フレームの視線推定結果を受け取るコールバック（消費側の出口）。
            指定時のみ毎フレーム呼ばれる。None なら従来どおり描画のみ。
            ロボット制御など外部アプリは gaze360 を import し、ここに処理を渡す。
        """
        self.source = source
        self.display_scale = display_scale
        self._on_results = on_results
        self._mjpeg_server = (
            MJPEGServer(stream_port, quality=stream_quality) if stream_port else None
        )

        print("=" * 50)
        print("  gaze360 パイプライン 初期化")
        print("=" * 50)
        self.detector = PersonDetector(model_size="n")
        self.head_detector = HeadDetector(model_size="n")
        self.estimator = GazeEstimator()
        self._fps = 0.0
        os.makedirs("outputs", exist_ok=True)

    # ------------------------------------------------------------------
    # メインループ
    # ------------------------------------------------------------------

    def run(self):
        """パイプラインを起動してメインループを実行する。"""
        if self._mjpeg_server:
            self._mjpeg_server.start()
            print("\n起動完了。Ctrl+C で終了\n")
        else:
            print("\n起動完了。'q' で終了、's' でフレーム保存\n")

        try:
            if self.source is None:
                self._run_camera()
            else:
                self._run_video(self.source)
        finally:
            # 正常終了・例外・Ctrl+C どの場合もソケットを確実に解放する
            if self._mjpeg_server:
                self._mjpeg_server.stop()

    def _run_camera(self):
        try:
            with ThetaStream() as stream:
                while True:
                    frame = stream.read()
                    if frame is None:
                        print("フレーム取得失敗。ストリームが終了しました。")
                        break
                    if not self._step(frame):
                        break
        except KeyboardInterrupt:
            print("\n終了します...")
        finally:
            if not self._mjpeg_server:
                cv2.destroyAllWindows()

    def _run_video(self, path):
        cap = cv2.VideoCapture(path)
        # HTTP ストリーム受信時は最新フレームだけ使うためバッファを最小化
        if path.startswith("http"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            if path.startswith("http"):
                raise ConnectionError(
                    f"HTTP ストリームに接続できません: {path}\n"
                    "Mac のリレーサーバーが起動しているか確認してください。"
                )
            raise FileNotFoundError(f"動画ファイルを開けません: {path}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if not self._step(frame):
                    break
        except KeyboardInterrupt:
            print("\n終了します...")
        finally:
            cap.release()
            if not self._mjpeg_server:
                cv2.destroyAllWindows()

    def _step(self, frame):
        """1フレーム処理して表示を更新。False を返したら終了。"""
        t0 = time.perf_counter()

        annotated, logs, results = self._process(frame)

        elapsed = time.perf_counter() - t0
        self._fps = 0.85 * self._fps + 0.15 * (1.0 / max(elapsed, 1e-6))

        for log in logs:
            print(log)

        # 消費側（ロボット制御など）への出口。指定時のみ呼ぶ。
        if self._on_results is not None:
            self._on_results(results)

        display = self._resize_for_display(annotated)

        if self._mjpeg_server:
            # ストリームモード: MJPEG サーバーにフレームを渡す（Ctrl+C で終了）
            self._mjpeg_server.update(display)
        else:
            # ローカルモード: cv2 ウィンドウ表示（従来の動作）
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
        results = []      # list[GazeResult] — 外部出力契約
        head_boxes = []   # 描画用の頭部枠の対角2隅 (corner_a, corner_b) | None（契約には含めない）

        for i, det in enumerate(detections):
            patch_rgb = equirect_to_perspective(
                frame,
                yaw_deg=det.yaw_deg,
                pitch_deg=det.pitch_deg,
                fov_deg=FOV_DEG,
                out_size=PATCH_SIZE,
            )
            # パッチ内で頭部 BBox を検出して Gaze-LLE に渡す（None ならフォールバック）
            head_bbox = self.head_detector.detect_head(patch_rgb)
            heatmap, inout = self.estimator.predict(patch_rgb, bbox=head_bbox)
            gaze_yaw, gaze_pitch = heatmap_to_spherical(
                heatmap,
                yaw_deg=det.yaw_deg,
                pitch_deg=det.pitch_deg,
                fov_deg=FOV_DEG,
            )

            # 頭部 BBox（パッチ正規化座標）の中心と対角2隅をワールド球面座標へ変換。
            # 中心は GazeResult.head_* と矢印起点に、2隅は頭部枠の描画に使う。
            head = self._head_to_spherical(head_bbox, det)
            if head is not None:
                (head_yaw, head_pitch), corner_a, corner_b = head
                head_boxes.append((corner_a, corner_b))
            else:
                head_yaw = head_pitch = None
                head_boxes.append(None)

            results.append(GazeResult(
                person_id=i + 1,
                gaze_yaw=gaze_yaw,
                gaze_pitch=gaze_pitch,
                inout=inout,
                confidence=det.confidence,
                head_yaw=head_yaw,
                head_pitch=head_pitch,
            ))

            status = "フレーム内" if inout >= INOUT_THRESH else "フレーム外"
            head_mark = "○" if head_bbox is not None else "×"
            logs.append(
                f"人物{i + 1}: 方位角 {gaze_yaw:+.1f}°, 仰角 {gaze_pitch:+.1f}° を見ている"
                f"  [{status}  inout={inout:.2f}  conf={det.confidence:.2f}  head={head_mark}]"
            )

        if not detections:
            logs.append(f"[フレーム] 人物なし  FPS={self._fps:.1f}")

        annotated = self._draw(frame, detections, results, head_boxes)
        return annotated, logs, results

    @staticmethod
    def _head_to_spherical(head_bbox, det):
        """頭部 BBox（パッチ正規化座標）を球面座標に変換する。

        Returns
        -------
        tuple | None
            ((中心az, 中心el), (隅A az, 隅A el), (隅B az, 隅B el))。
            頭部が検出されていなければ None。
        """
        if head_bbox is None:
            return None
        x1n, y1n, x2n, y2n = head_bbox
        center = patch_norm_to_spherical(
            (x1n + x2n) / 2, (y1n + y2n) / 2, det.yaw_deg, det.pitch_deg, FOV_DEG
        )
        corner_a = patch_norm_to_spherical(x1n, y1n, det.yaw_deg, det.pitch_deg, FOV_DEG)
        corner_b = patch_norm_to_spherical(x2n, y2n, det.yaw_deg, det.pitch_deg, FOV_DEG)
        return center, corner_a, corner_b

    # ------------------------------------------------------------------
    # 可視化
    # ------------------------------------------------------------------

    def _draw(self, frame, detections, results, head_boxes):
        vis = frame.copy()
        h, w = vis.shape[:2]

        for i, (det, res, head_box) in enumerate(
            zip(detections, results, head_boxes)
        ):
            x1, y1, x2, y2 = (int(v) for v in det.bbox_px)

            # 人物 BBox（緑）
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_PERSON, 2)
            cv2.putText(
                vis, f"P{i + 1}",
                (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_PERSON, 2,
            )

            # 360度境界マージ時は2片目（反対側の端）も描く
            if det.extra_bbox_px is not None:
                ex1, ey1, ex2, ey2 = (int(v) for v in det.extra_bbox_px)
                cv2.rectangle(vis, (ex1, ey1), (ex2, ey2), COLOR_PERSON, 2)
                cv2.putText(
                    vis, f"P{i + 1}",
                    (ex1, max(ey1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_PERSON, 2,
                )

            # 矢印起点: 頭部中心（検出時）。頭部 BBox はマゼンタで描画する。
            # 未検出時は人物の球面中心へフォールバック（det.yaw_deg はマージ人物でも
            # つなぎ目＝真の位置を指すため、体の左片中心より起点が正確になる）。
            if res.head_yaw is not None:
                origin_yaw, origin_pitch = res.head_yaw, res.head_pitch
                if head_box is not None:
                    corner_a, corner_b = head_box
                    _draw_head_box(vis, corner_a, corner_b, COLOR_HEAD, w, h)
            else:
                origin_yaw, origin_pitch = det.yaw_deg, det.pitch_deg
            ox, oy = _spherical_to_pixel(origin_yaw, origin_pitch, w, h)

            # 視線先ピクセル
            gx, gy = _spherical_to_pixel(res.gaze_yaw, res.gaze_pitch, w, h)
            color = COLOR_GAZE_IN if res.inout >= INOUT_THRESH else COLOR_GAZE_OUT

            # 矢印（頭部起点→視線先。境界をまたぐ場合は2分割で画面横断を防ぐ）
            _draw_gaze_arrow(vis, ox, oy, origin_yaw, res.gaze_yaw, gx, gy, color, w)

            # 視線先マーカー（塗りつぶし円）
            cv2.circle(vis, (gx, gy), 10, color, -1)
            cv2.circle(vis, (gx, gy), 10, (255, 255, 255), 1)  # 白縁

            # 方位角・仰角テキスト
            label = f"az:{res.gaze_yaw:+.0f} el:{res.gaze_pitch:+.0f}"
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
        "--scale", type=float, default=None,
        help=f"表示・配信フレームの縮小倍率（省略時: ローカル {DISPLAY_SCALE} / "
             f"ストリーム {STREAM_SCALE}）",
    )
    parser.add_argument(
        "--stream-port", type=int, default=None, dest="stream_port",
        help="MJPEG HTTP ストリームのポート番号（例: 8080）。"
             "指定時はブラウザで映像を視聴。省略時は cv2 ウィンドウ表示。",
    )
    parser.add_argument(
        "--stream-quality", type=int, default=STREAM_JPEG_QUALITY, dest="stream_quality",
        help=f"ストリーム配信時の JPEG 品質 1-100（デフォルト: {STREAM_JPEG_QUALITY}）。"
             "高いほど高画質・高帯域。VPN が細い場合は下げる。",
    )
    args = parser.parse_args()

    # scale 未指定時: ストリーム配信は帯域節約のため縮小、ローカルは原寸
    if args.scale is not None:
        scale = args.scale
    elif args.stream_port:
        scale = STREAM_SCALE
    else:
        scale = DISPLAY_SCALE

    pipeline = GazePipeline(
        source=args.source,
        display_scale=scale,
        stream_port=args.stream_port,
        stream_quality=args.stream_quality,
    )

    # SSH 切断（SIGHUP）や終了シグナル（SIGTERM）でソケットを確実に解放する
    def _handle_signal(signum, frame):
        print(f"\nシグナル {signum} を受信。終了します...")
        if pipeline._mjpeg_server:
            pipeline._mjpeg_server.stop()
        sys.exit(0)

    signal.signal(signal.SIGHUP, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    pipeline.run()


if __name__ == "__main__":
    main()
