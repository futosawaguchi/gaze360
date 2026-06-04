"""
フェーズ3 ラッパーの動作確認スクリプト。

テスト1: GazeEstimator のインポートと推論（モデルロードあり）
テスト2: ThetaStream のインポートと接続設定の確認（カメラ不要）
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def test_gaze_estimator():
    print("=== テスト1: GazeEstimator ===")
    from src.gaze.estimator import GazeEstimator
    from src.projection.equirect import equirect_to_perspective

    estimator = GazeEstimator()

    # 合成 equirectangular フレームから透視投影パッチを切り出して推論
    np.random.seed(0)
    frame_bgr = np.random.randint(0, 256, (960, 1920, 3), dtype=np.uint8)
    patch_rgb = equirect_to_perspective(frame_bgr, yaw_deg=0.0, pitch_deg=0.0)

    heatmap, inout = estimator.predict(patch_rgb)

    assert heatmap.shape == (64, 64), f"想定外のshape: {heatmap.shape}"
    assert heatmap.dtype == np.float32, f"想定外のdtype: {heatmap.dtype}"
    assert 0.0 <= inout <= 1.0, f"inout が範囲外: {inout}"
    print(f"  heatmap.shape : {heatmap.shape}")
    print(f"  heatmap 最大値: {heatmap.max():.4f}")
    print(f"  inout スコア  : {inout:.4f}")
    print("  PASS\n")
    return True


def test_theta_stream_import():
    print("=== テスト2: ThetaStream インポート・設定確認 ===")
    from src.camera.theta_stream import ThetaStream

    stream = ThetaStream()

    # open() は実際のカメラがないと呼べないが、
    # 接続情報のビルドが正しいかだけ確認する
    url = ThetaStream._get_url()
    assert url.startswith("http://"), f"想定外の URL: {url}"
    print(f"  接続先 URL: {url}")

    # THETA_SERIAL 未設定の環境では EnvironmentError が出ることを確認
    if not os.getenv("THETA_SERIAL"):
        try:
            ThetaStream._get_auth()
            assert False, "EnvironmentError が発生しなかった"
        except EnvironmentError as e:
            print(f"  THETA_SERIAL 未設定 → 想定通り EnvironmentError: {e}")
    else:
        auth = ThetaStream._get_auth()
        print(f"  認証情報取得 OK (username={auth.username})")

    print("  PASS\n")
    return True


if __name__ == "__main__":
    ok1 = test_gaze_estimator()
    ok2 = test_theta_stream_import()

    if ok1 and ok2:
        print("全テスト PASS")
        sys.exit(0)
    else:
        print("テスト失敗あり")
        sys.exit(1)
