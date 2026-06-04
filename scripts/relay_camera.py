"""
THETA X MJPEG リレーサーバー（Mac 上で実行する）

THETA X からフレームを取得し、HTTP MJPEG ストリームとして LAN に配信する。
GPU サーバーは http://<MacのIP>:<PORT>/stream で受信できる。

使い方:
    python scripts/relay_camera.py
    python scripts/relay_camera.py --port 9090
"""
import argparse
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from src.camera.theta_stream import ThetaStream

DEFAULT_PORT = 9090


class _RelayHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/stream":
            self._send_stream()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>THETA X Relay — use /stream</body></html>")

    def _send_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        relay = self.server.relay
        try:
            while True:
                with relay.lock:
                    frame = relay.latest_frame
                if frame is not None:
                    ok, jpeg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )
                    if ok:
                        data = jpeg.tobytes()
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + data
                            + b"\r\n"
                        )
                time.sleep(0.033)  # ~30FPS 上限
        except Exception:
            pass

    def log_message(self, *args):
        pass  # アクセスログを抑制


class RelayServer:
    def __init__(self, port):
        self.port = port
        self.latest_frame = None
        self.lock = threading.Lock()
        self._server = None

    def update(self, frame_bgr):
        with self.lock:
            self.latest_frame = frame_bgr.copy()

    def start(self):
        HTTPServer.allow_reuse_address = True
        server = HTTPServer(("0.0.0.0", self.port), _RelayHandler)
        server.relay = self
        self._server = server
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[RelayServer] ポート {self.port} で配信開始")
        print(f"[RelayServer] GPU サーバーから: http://<Mac の IP>:{self.port}/stream")

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def main():
    parser = argparse.ArgumentParser(description="THETA X MJPEG リレーサーバー")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="配信ポート番号")
    args = parser.parse_args()

    relay = RelayServer(args.port)
    relay.start()

    print("[Relay] THETA X に接続中...")
    try:
        with ThetaStream() as stream:
            print("[Relay] 配信中。Ctrl+C で終了\n")
            while True:
                frame = stream.read()
                if frame is None:
                    print("[Relay] フレーム取得失敗。終了します。")
                    break
                relay.update(frame)
    except KeyboardInterrupt:
        print("\n[Relay] 終了します...")
    finally:
        relay.stop()


if __name__ == "__main__":
    main()
