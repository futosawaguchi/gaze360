import os
import threading
import time

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

_OSC_ENDPOINT = "/osc/commands/execute"
_LIVE_PREVIEW_CMD = "camera.getLivePreview"
_JPEG_SOI = b"\xff\xd8"  # Start of Image
_JPEG_EOI = b"\xff\xd9"  # End of Image


class ThetaStream:
    """RICOH THETA X のライブプレビューストリームをラップするクラス。

    バックグラウンドスレッドで MJPEG を常時読み続け、最新フレームだけを保持する。
    THETA X は約15FPSで送出し続けるため、消費側（推論）が遅いとフレームが
    受信バッファに溜まって遅延が蓄積する。最新フレームだけを返すことで
    遅延蓄積を防ぐ（古いフレームは捨てる）。

    .env ファイルに以下の変数を設定すること:
        THETA_SERIAL  : カメラのシリアル番号（必須）
        THETA_IP      : カメラの IP アドレス（省略時 192.168.1.1）
        THETA_PASSWORD: Wi-Fi パスワード（省略時はシリアル番号と同じ）

    Usage:
        stream = ThetaStream()
        stream.open()
        frame = stream.read()   # BGR uint8 numpy array (H, W, 3)
        stream.close()

        # または with 文で使用
        with ThetaStream() as stream:
            frame = stream.read()
    """

    def __init__(self, chunk_size=1024):
        """
        Parameters
        ----------
        chunk_size : int
            HTTP ストリームの読み取りチャンクサイズ（バイト）。
        """
        self._chunk_size = chunk_size
        self._response = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame = None
        self._frame_id = 0          # フレーム更新ごとにインクリメント
        self._last_returned_id = -1  # read() が最後に返したフレームID

    # ------------------------------------------------------------------
    # 接続管理
    # ------------------------------------------------------------------

    def open(self):
        """カメラに接続してストリームを開始し、読み取りスレッドを起動する。"""
        url = self._get_url()
        auth = self._get_auth()
        self._response = requests.post(
            url,
            json={"name": _LIVE_PREVIEW_CMD},
            auth=auth,
            stream=True,
            timeout=10,
        )
        if self._response.status_code != 200:
            raise ConnectionError(
                f"ライブプレビュー開始失敗 (HTTP {self._response.status_code})"
            )
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def close(self):
        """ストリームを閉じ、読み取りスレッドを停止する。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._response is not None:
            self._response.close()
            self._response = None
        with self._lock:
            self._latest_frame = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # 読み取りスレッド
    # ------------------------------------------------------------------

    def _reader_loop(self):
        """MJPEG ストリームを読み続け、最新フレームを保持する（別スレッド）。"""
        buf = b""
        try:
            for chunk in self._response.iter_content(chunk_size=self._chunk_size):
                if not self._running:
                    break
                buf += chunk
                start = buf.find(_JPEG_SOI)
                end = buf.find(_JPEG_EOI)
                if start != -1 and end != -1:
                    jpg = buf[start : end + 2]
                    buf = buf[end + 2 :]
                    frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is not None:
                        with self._lock:
                            self._latest_frame = frame
                            self._frame_id += 1
        except Exception:
            pass  # 接続断などはループ終了として扱う
        finally:
            self._running = False

    # ------------------------------------------------------------------
    # フレーム取得
    # ------------------------------------------------------------------

    def read(self, timeout=5.0):
        """最新フレームを取得する。

        新しいフレームが到着するまで待ち、最新フレームを返す。
        同じフレームを二重に処理しないよう、前回返したフレームと同じものは返さない
        （カメラの送出FPSが実効上限になる）。

        Parameters
        ----------
        timeout : float
            新フレームを待つ最大秒数。これを超えたらストリーム断とみなす。

        Returns
        -------
        frame : np.ndarray | None
            BGR uint8 equirectangular フレーム shape (H, W, 3)。
            タイムアウト（ストリーム断）した場合は None。
        """
        deadline = time.time() + timeout
        while True:
            with self._lock:
                if self._latest_frame is not None and self._frame_id != self._last_returned_id:
                    self._last_returned_id = self._frame_id
                    return self._latest_frame.copy()
            if not self._running and self._latest_frame is None:
                return None  # スレッドが停止し、一度もフレームが来ていない
            if time.time() > deadline:
                return None
            time.sleep(0.005)

    # ------------------------------------------------------------------
    # 接続情報の組み立て
    # ------------------------------------------------------------------

    @staticmethod
    def _get_url():
        ip = os.getenv("THETA_IP", "192.168.1.1")
        return f"http://{ip}{_OSC_ENDPOINT}"

    @staticmethod
    def _get_auth():
        serial = os.getenv("THETA_SERIAL")
        if not serial:
            raise EnvironmentError(
                "THETA_SERIAL が設定されていません。.env ファイルを確認してください。"
            )
        password = os.getenv("THETA_PASSWORD", serial)
        return requests.auth.HTTPDigestAuth(f"THETA{serial}", password)
