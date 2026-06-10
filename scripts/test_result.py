"""
src/gaze/result.py の動作確認スクリプト。

テスト1: GazeResult の構築（全フィールド指定／head_* 省略）
テスト2: select_primary（confidence 最大を返す／空リストは None）
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gaze.result import GazeResult, select_primary


def test_gaze_result_construction():
    print("=== テスト1: GazeResult 構築 ===")
    all_pass = True

    # 全フィールド指定（頭部検出あり）
    r = GazeResult(
        person_id=1, gaze_yaw=30.0, gaze_pitch=-10.0,
        inout=0.8, confidence=0.9, head_yaw=28.0, head_pitch=5.0,
    )
    ok = (r.person_id == 1 and r.gaze_yaw == 30.0 and r.head_yaw == 28.0)
    all_pass &= ok
    print(f"  {'PASS' if ok else 'FAIL'}: 全フィールド指定 {r}")

    # head_* 省略（頭部未検出）→ None がデフォルト
    r2 = GazeResult(person_id=2, gaze_yaw=0.0, gaze_pitch=0.0, inout=0.1, confidence=0.6)
    ok2 = (r2.head_yaw is None and r2.head_pitch is None)
    all_pass &= ok2
    print(f"  {'PASS' if ok2 else 'FAIL'}: head_* 省略時 None（{r2.head_yaw}, {r2.head_pitch}）")

    print("-> 全ケース PASS\n" if all_pass else "-> 失敗あり\n")
    return all_pass


def test_select_primary():
    print("=== テスト2: select_primary ===")
    all_pass = True

    results = [
        GazeResult(person_id=1, gaze_yaw=0.0, gaze_pitch=0.0, inout=0.5, confidence=0.6),
        GazeResult(person_id=2, gaze_yaw=10.0, gaze_pitch=0.0, inout=0.5, confidence=0.95),
        GazeResult(person_id=3, gaze_yaw=20.0, gaze_pitch=0.0, inout=0.5, confidence=0.8),
    ]
    primary = select_primary(results)
    ok = primary is not None and primary.person_id == 2  # confidence 最大
    all_pass &= ok
    print(f"  {'PASS' if ok else 'FAIL'}: confidence 最大を選択（選=P{primary.person_id if primary else None}, 期待 P2）")

    # 空リスト → None
    ok_empty = select_primary([]) is None
    all_pass &= ok_empty
    print(f"  {'PASS' if ok_empty else 'FAIL'}: 空リスト→None")

    print("-> 全ケース PASS\n" if all_pass else "-> 失敗あり\n")
    return all_pass


if __name__ == "__main__":
    ok1 = test_gaze_result_construction()
    ok2 = test_select_primary()

    if ok1 and ok2:
        print("全テスト PASS")
        sys.exit(0)
    else:
        print("テスト失敗あり")
        sys.exit(1)
