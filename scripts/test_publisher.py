"""
src/gaze/publisher.py の動作確認スクリプト（モデル不要・ループバック）。

テスト1: publish した list[GazeResult] が TCP 経由で1行 JSON として受信でき、
         フィールドが復元できること。
テスト2: 人物ゼロ（空リスト）が [] として配信されること。
"""
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gaze.publisher import GazeResultPublisher
from src.gaze.result import GazeResult

_PORT = 8390  # テスト用ポート（既定の配信ポート例 8090 とは別にする）


def _recv_one_line(sock, timeout=3.0):
    sock.settimeout(timeout)
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf.split(b"\n", 1)[0]


def test_publish_roundtrip():
    print("=== テスト1: publish → TCP 受信 → JSON 復元 ===")
    pub = GazeResultPublisher(_PORT)
    pub.start()
    all_pass = True
    try:
        client = socket.create_connection(("127.0.0.1", _PORT), timeout=3.0)
        # クライアント接続後に publish（最新行が配信される）
        time.sleep(0.05)
        results = [
            GazeResult(person_id=1, gaze_yaw=30.0, gaze_pitch=-10.0,
                       inout=0.8, confidence=0.9, head_yaw=28.0, head_pitch=5.0),
            GazeResult(person_id=2, gaze_yaw=-90.0, gaze_pitch=0.0,
                       inout=0.2, confidence=0.6),
        ]
        pub.publish(results)

        line = _recv_one_line(client)
        data = json.loads(line.decode("utf-8"))
        ok = (
            isinstance(data, list) and len(data) == 2
            and data[0]["person_id"] == 1
            and abs(data[0]["gaze_yaw"] - 30.0) < 1e-6
            and data[0]["head_yaw"] == 28.0
            and data[1]["head_yaw"] is None      # 頭部未検出は null
        )
        all_pass &= ok
        print(f"  {'PASS' if ok else 'FAIL'}: 2件受信・フィールド復元 {data}")
        client.close()
    finally:
        pub.stop()

    print("-> 全ケース PASS\n" if all_pass else "-> 失敗あり\n")
    return all_pass


def test_publish_empty():
    print("=== テスト2: 人物ゼロ → [] 配信 ===")
    pub = GazeResultPublisher(_PORT)
    pub.start()
    all_pass = True
    try:
        client = socket.create_connection(("127.0.0.1", _PORT), timeout=3.0)
        time.sleep(0.05)
        pub.publish([])
        line = _recv_one_line(client)
        data = json.loads(line.decode("utf-8"))
        ok = data == []
        all_pass &= ok
        print(f"  {'PASS' if ok else 'FAIL'}: 空フレーム配信 {data!r}")
        client.close()
    finally:
        pub.stop()

    print("-> 全ケース PASS\n" if all_pass else "-> 失敗あり\n")
    return all_pass


if __name__ == "__main__":
    ok1 = test_publish_roundtrip()
    ok2 = test_publish_empty()

    if ok1 and ok2:
        print("全テスト PASS")
        sys.exit(0)
    else:
        print("テスト失敗あり")
        sys.exit(1)
