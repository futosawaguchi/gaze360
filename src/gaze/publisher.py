"""
GazeResult を行区切り JSON で TCP 配信する任意 sink。

疎結合モード（GPU で知覚 → ローカルが購読 → ロボット制御）向けの「視線データの出口」。
gaze360 本体は Sota を知らず、汎用の視線データを配信するだけ。

設計は pipeline.MJPEGServer に倣う:
  - バックグラウンドスレッドで accept
  - クライアント毎にハンドラスレッド
  - 「最新フレームのみ」を配信（遅い購読者がパイプラインを止めない／バックログを溜めない。
    ThetaStream の最新フレーム方針と同じ思想）

transport は TCP（SSH -L で素直に転送できる。UDP は SSH 転送が面倒なため避ける）。
配信フォーマット（1行 = 1フレーム、改行区切り）:
  [{"person_id":1,"gaze_yaw":..,"gaze_pitch":..,"inout":..,"confidence":..,
    "head_yaw":..,"head_pitch":..}, ...]\\n
人物ゼロのフレームは [] を配信する。

Usage:
    pub = GazeResultPublisher(8090)
    pub.start()
    pub.publish(results)   # list[GazeResult]
    pub.stop()
"""

import json
import socket
import threading
import time
from dataclasses import asdict


class GazeResultPublisher:
    """GazeResult を行区切り JSON で TCP 配信するバックグラウンドサーバー。"""

    def __init__(self, port: int):
        self.port = port
        self._server = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_line = None   # 最新の配信行（bytes）
        self._seq = 0              # publish のたびに増加（最新判定用）

    def start(self):
        """TCP サーバーをバックグラウンドスレッドで起動する。"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 前回終了直後でも同じポートで再起動できるように
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.port))
        server.listen(5)
        self._server = server
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()
        print(f"[GazeResultPublisher] TCP {self.port} で視線データを配信中（行区切りJSON）")

    def _accept_loop(self):
        while self._running:
            try:
                client, _ = self._server.accept()
            except OSError:
                break  # stop() でソケットが閉じられた
            threading.Thread(
                target=self._client_loop, args=(client,), daemon=True
            ).start()

    def _client_loop(self, client):
        """接続中クライアントへ、最新行が更新されるたびに送る。"""
        last_sent = -1
        client.settimeout(5.0)
        try:
            while self._running:
                with self._lock:
                    line, seq = self._latest_line, self._seq
                if line is not None and seq != last_sent:
                    client.sendall(line)
                    last_sent = seq
                else:
                    time.sleep(0.005)
        except OSError:
            pass  # 切断・送信エラー
        finally:
            client.close()

    def publish(self, results):
        """最新の視線結果を配信用にセットする（パイプラインのメインループから毎フレーム呼ぶ）。

        Parameters
        ----------
        results : list[GazeResult]
            1フレーム分の視線推定結果。空でも [] を配信する。
        """
        line = (json.dumps([asdict(r) for r in results]) + "\n").encode("utf-8")
        with self._lock:
            self._latest_line = line
            self._seq += 1

    def stop(self):
        self._running = False
        if self._server is not None:
            self._server.close()  # accept をほどいてスレッドを終了させる
            self._server = None
