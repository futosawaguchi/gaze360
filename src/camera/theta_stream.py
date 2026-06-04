import os

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
        self._iter = None
        self._buf = b""

    # ------------------------------------------------------------------
    # 接続管理
    # ------------------------------------------------------------------

    def open(self):
        """カメラに接続してストリームを開始する。"""
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
        self._iter = self._response.iter_content(chunk_size=self._chunk_size)
        self._buf = b""

    def close(self):
        """ストリームを閉じる。"""
        if self._response is not None:
            self._response.close()
            self._response = None
            self._iter = None
            self._buf = b""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # フレーム取得
    # ------------------------------------------------------------------

    def read(self):
        """次のフレームを取得する。

        MJPEG ストリームから JPEG 1 枚を切り出し、BGR numpy 配列として返す。
        フレームが取得できない場合は None を返す。

        Returns
        -------
        frame : np.ndarray | None
            BGR uint8 equirectangular フレーム shape (H, W, 3)。
            ストリームが終了した場合は None。
        """
        if self._iter is None:
            raise RuntimeError("ストリームが開いていません。open() を先に呼んでください。")

        for chunk in self._iter:
            self._buf += chunk
            start = self._buf.find(_JPEG_SOI)
            end = self._buf.find(_JPEG_EOI)
            if start != -1 and end != -1:
                jpg = self._buf[start : end + 2]
                self._buf = self._buf[end + 2 :]
                frame = cv2.imdecode(
                    np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    return frame
        return None

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
