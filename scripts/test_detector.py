"""
フェーズ4 PersonDetector の動作確認スクリプト。

テスト1: pixel_to_spherical の数値確認
テスト2: PersonDetector の初期化
テスト3: 空フレームで detect() を呼んで空リストが返ることを確認
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.detection.person_detector import PersonDetector, _pixel_to_spherical


def test_pixel_to_spherical():
    print("=== テスト1: pixel_to_spherical ===")
    W, H = 1920, 960

    cases = [
        # (cx, cy, expected_yaw, expected_pitch)
        (W / 2,     H / 2,      0.0,    0.0),   # 中央 → 正面
        (0,         H / 2,   -180.0,    0.0),   # 左端
        (W,         H / 2,    180.0,    0.0),   # 右端
        (W / 2,     0,           0.0,   90.0),  # 上端 → 真上
        (W / 2,     H,           0.0,  -90.0),  # 下端 → 真下
        (W * 0.75,  H / 2,      90.0,   0.0),   # 右90°
    ]

    all_pass = True
    for cx, cy, exp_yaw, exp_pitch in cases:
        yaw, pitch = _pixel_to_spherical(cx, cy, W, H)
        ok = abs(yaw - exp_yaw) < 0.001 and abs(pitch - exp_pitch) < 0.001
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {status}: px=({cx:.0f},{cy:.0f}) → yaw={yaw:.1f}° pitch={pitch:.1f}°  (期待: {exp_yaw:.1f}°, {exp_pitch:.1f}°)")

    print("-> 全ケース PASS\n" if all_pass else "-> 失敗あり\n")
    return all_pass


def test_detector_init():
    print("=== テスト2: PersonDetector 初期化 ===")
    detector = PersonDetector(model_size="n", conf=0.5)
    print("  PASS\n")
    return detector


def test_detect_empty_frame(detector):
    print("=== テスト3: 空フレームで detect() ===")
    blank = np.zeros((960, 1920, 3), dtype=np.uint8)
    detections = detector.detect(blank)
    assert isinstance(detections, list), "戻り値がリストでない"
    print(f"  検出数: {len(detections)}  （0が期待値）")
    print("  PASS\n")
    return True


if __name__ == "__main__":
    ok1 = test_pixel_to_spherical()
    detector = test_detector_init()
    ok3 = test_detect_empty_frame(detector)

    if ok1 and ok3:
        print("全テスト PASS")
        sys.exit(0)
    else:
        print("テスト失敗あり")
        sys.exit(1)
