"""
PersonDetector / HeadDetector の動作確認スクリプト。

テスト1: pixel_to_spherical の数値確認
テスト2: PersonDetector の初期化
テスト3: 空フレームで detect() を呼んで空リストが返ることを確認
テスト4: _merge_wraparound（360度境界マージ）の確認
テスト5: HeadDetector の頭部BBox構成の確認
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.detection.person_detector import (
    PersonDetector,
    _pixel_to_spherical,
    _merge_wraparound,
)
from src.detection.head_detector import HeadDetector


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


def test_merge_wraparound():
    print("=== テスト4: _merge_wraparound（360度境界マージ）===")
    W, H = 1920, 960
    all_pass = True

    # ケースA: 左端片 + 右端片（y重なりあり）→ 1人に統合
    boxes = [
        (0.0, 100.0, 50.0, 300.0, 0.8),       # 左端接触
        (1870.0, 110.0, 1920.0, 310.0, 0.7),  # 右端接触
    ]
    merged = _merge_wraparound(boxes, W, H)
    ok = len(merged) == 1 and merged[0][4] is not None
    if ok:
        cx_for_yaw, cy, primary, conf, extra = merged[0]
        yaw, _ = _pixel_to_spherical(cx_for_yaw, cy, W, H)
        ok_yaw = abs(abs(yaw) - 180.0) < 20.0  # つなぎ目付近 ±180°
        ok = ok and ok_yaw
        print(f"  {'PASS' if ok else 'FAIL'}: 左右端→1人に統合, yaw={yaw:.1f}°（±180°付近）")
    else:
        print(f"  FAIL: 統合されず（件数={len(merged)}）")
    all_pass &= ok

    # ケースB: 端に接しない2つ → 統合されない
    boxes_b = [
        (100.0, 100.0, 200.0, 300.0, 0.9),
        (800.0, 100.0, 900.0, 300.0, 0.8),
    ]
    merged_b = _merge_wraparound(boxes_b, W, H)
    ok_b = len(merged_b) == 2 and all(m[4] is None for m in merged_b)
    all_pass &= ok_b
    print(f"  {'PASS' if ok_b else 'FAIL'}: 中央2人→統合されない（件数={len(merged_b)}）")

    # ケースC: 左端+右端だが y が重ならない → 別人物
    boxes_c = [
        (0.0, 50.0, 50.0, 200.0, 0.8),
        (1870.0, 500.0, 1920.0, 700.0, 0.7),
    ]
    merged_c = _merge_wraparound(boxes_c, W, H)
    ok_c = len(merged_c) == 2
    all_pass &= ok_c
    print(f"  {'PASS' if ok_c else 'FAIL'}: 左右端だがy重ならない→別人物（件数={len(merged_c)}）")

    print("-> 全ケース PASS\n" if all_pass else "-> 失敗あり\n")
    return all_pass


def test_head_bbox_construction():
    print("=== テスト5: HeadDetector 頭部BBox構成 ===")
    all_pass = True
    W, H = 448, 448

    # 顔 keypoint（鼻/両目/両耳）が中央付近にある合成データ
    # kps: (17, 3) = (x, y, conf)。顔まわり 0-4 のみ可視にする
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[0] = [224, 200, 0.9]  # 鼻
    kps[1] = [210, 185, 0.9]  # 左目
    kps[2] = [238, 185, 0.9]  # 右目
    kps[3] = [195, 190, 0.8]  # 左耳
    kps[4] = [253, 190, 0.8]  # 右耳

    bbox = HeadDetector._face_kps_to_head_bbox(kps, W, H)
    ok = (
        bbox is not None
        and all(0.0 <= v <= 1.0 for v in bbox)
        and bbox[0] < bbox[2]
        and bbox[1] < bbox[3]
    )
    all_pass &= ok
    if bbox is not None:
        print(f"  {'PASS' if ok else 'FAIL'}: 頭部BBox(正規化)={tuple(round(v, 3) for v in bbox)}")
    else:
        print("  FAIL: bbox が None")

    # 全 keypoint が低信頼度 → None
    kps_low = np.zeros((17, 3), dtype=np.float32)  # conf 全て 0
    bbox_none = HeadDetector._face_kps_to_head_bbox(kps_low, W, H)
    ok_none = bbox_none is None
    all_pass &= ok_none
    print(f"  {'PASS' if ok_none else 'FAIL'}: 低信頼度→None")

    print("-> 全ケース PASS\n" if all_pass else "-> 失敗あり\n")
    return all_pass


if __name__ == "__main__":
    ok1 = test_pixel_to_spherical()
    detector = test_detector_init()
    ok3 = test_detect_empty_frame(detector)
    ok4 = test_merge_wraparound()
    ok5 = test_head_bbox_construction()

    if ok1 and ok3 and ok4 and ok5:
        print("全テスト PASS")
        sys.exit(0)
    else:
        print("テスト失敗あり")
        sys.exit(1)
