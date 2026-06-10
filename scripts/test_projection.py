"""
src/projection/equirect.py の動作確認スクリプト。

テスト1: heatmap_to_spherical の数値ラウンドトリップ
  既知の球面座標 → 対応するピクセルにGaussianヒートマップを生成
  → heatmap_to_spherical で復元 → 誤差 < 0.01° を確認

テスト2: equirect_to_perspective のスモークテスト
  合成Equirectangular画像（ランダムノイズ）で e2p が正しく動くか確認

テスト3: patch_norm_to_spherical の変換確認
  パッチ中心(0.5,0.5)→カメラ方向、および heatmap 経路と同じ正規化で
  既知の球面座標を復元できることを確認（誤差 < 0.01°）
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.projection.equirect import (
    equirect_to_perspective,
    heatmap_to_spherical,
    patch_norm_to_spherical,
)


def make_gaussian_heatmap(peak_row, peak_col, H=64, W=64, sigma=2.0):
    rows = np.arange(H, dtype=np.float64).reshape(H, 1)
    cols = np.arange(W, dtype=np.float64).reshape(1, W)
    heatmap = np.exp(-((rows - peak_row) ** 2 + (cols - peak_col) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


def world_dir_to_heatmap_pixel(target_az, target_el, cam_yaw, cam_pitch, fov_deg, H=64, W=64):
    """球面座標をヒートマップのピクセル座標に変換（順方向）。"""
    # ワールド方向 → 直交座標
    az_r = np.deg2rad(target_az)
    el_r = np.deg2rad(target_el)
    world_xyz = np.array([
        np.cos(el_r) * np.sin(az_r),
        np.sin(el_r),
        np.cos(el_r) * np.cos(az_r),
    ])

    # py360convert と同じ回転の逆：world → cam = (Rx @ Ry)^T @ world
    u = -np.deg2rad(cam_yaw)
    v = np.deg2rad(cam_pitch)
    cos_u, sin_u = np.cos(u), np.sin(u)
    cos_v, sin_v = np.cos(v), np.sin(v)
    Rx = np.array([[1, 0, 0], [0, cos_v, -sin_v], [0, sin_v, cos_v]])
    Ry = np.array([[cos_u, 0, sin_u], [0, 1, 0], [-sin_u, 0, cos_u]])
    # equirect.py は右乗算: world = cam @ Rx @ Ry
    # 逆変換: cam = world @ Ry.T @ Rx.T
    cam_xyz = world_xyz @ Ry.T @ Rx.T

    tan_half = np.tan(np.deg2rad(fov_deg / 2.0))
    x_tan = cam_xyz[0] / cam_xyz[2]
    y_tan = cam_xyz[1] / cam_xyz[2]
    col_f = (x_tan + tan_half) / (2 * tan_half) * (W - 1)
    row_f = (-y_tan + tan_half) / (2 * tan_half) * (H - 1)
    return row_f, col_f


def test_round_trip():
    print("=== テスト1: heatmap_to_spherical ラウンドトリップ ===")

    cases = [
        # (target_az, target_el, cam_yaw, cam_pitch, fov_deg)
        (50.0, 25.0, 30.0, 15.0, 90.0),
        (0.0, 0.0, 0.0, 0.0, 90.0),      # 中心
        (-90.0, 0.0, -90.0, 0.0, 90.0),  # 真左
        (10.0, -20.0, 20.0, -10.0, 60.0),
        (170.0, 5.0, 170.0, 5.0, 90.0),  # 境界付近
    ]

    all_pass = True
    for target_az, target_el, cam_yaw, cam_pitch, fov_deg in cases:
        row_f, col_f = world_dir_to_heatmap_pixel(
            target_az, target_el, cam_yaw, cam_pitch, fov_deg
        )

        # ピクセルがヒートマップ内に収まる場合のみテスト
        if not (0 <= row_f <= 63 and 0 <= col_f <= 63):
            print(f"  SKIP: ({target_az:.1f}°, {target_el:.1f}°) はパッチ外 (row={row_f:.1f}, col={col_f:.1f})")
            continue

        heatmap = make_gaussian_heatmap(row_f, col_f)
        az_out, el_out = heatmap_to_spherical(heatmap, cam_yaw, cam_pitch, fov_deg)

        az_err = abs(az_out - target_az)
        el_err = abs(el_out - target_el)
        passed = az_err < 0.01 and el_err < 0.01
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(
            f"  {status}: target=({target_az:.1f}°, {target_el:.1f}°) "
            f"recovered=({az_out:.4f}°, {el_out:.4f}°) "
            f"err=(az:{az_err:.5f}°, el:{el_err:.5f}°)"
        )

    if all_pass:
        print("-> 全ケース PASS\n")
    else:
        print("-> 失敗あり\n")
    return all_pass


def test_patch_norm_to_spherical():
    print("=== テスト3: patch_norm_to_spherical 変換 ===")
    all_pass = True

    # ケースA: パッチ中心 (0.5, 0.5) はカメラ方向そのものを返す
    center_cases = [
        # (cam_yaw, cam_pitch, fov_deg)
        (30.0, 15.0, 90.0),
        (-90.0, 0.0, 90.0),
        (170.0, 5.0, 60.0),
    ]
    for cam_yaw, cam_pitch, fov_deg in center_cases:
        az, el = patch_norm_to_spherical(0.5, 0.5, cam_yaw, cam_pitch, fov_deg)
        ok = abs(az - cam_yaw) < 0.01 and abs(el - cam_pitch) < 0.01
        all_pass &= ok
        print(
            f"  {'PASS' if ok else 'FAIL'}: 中心(0.5,0.5) cam=({cam_yaw:.1f}°,{cam_pitch:.1f}°) "
            f"→ ({az:.4f}°, {el:.4f}°)"
        )

    # ケースB: heatmap_to_spherical と同じ格子・正規化で同じ結果になることを確認
    grid_cases = [
        # (target_az, target_el, cam_yaw, cam_pitch, fov_deg)
        (50.0, 25.0, 30.0, 15.0, 90.0),
        (10.0, -20.0, 20.0, -10.0, 60.0),
    ]
    H = W = 64
    for target_az, target_el, cam_yaw, cam_pitch, fov_deg in grid_cases:
        row_f, col_f = world_dir_to_heatmap_pixel(
            target_az, target_el, cam_yaw, cam_pitch, fov_deg, H, W
        )
        if not (0 <= row_f <= H - 1 and 0 <= col_f <= W - 1):
            print(f"  SKIP: ({target_az:.1f}°, {target_el:.1f}°) はパッチ外")
            continue
        az, el = patch_norm_to_spherical(
            col_f / (W - 1), row_f / (H - 1), cam_yaw, cam_pitch, fov_deg
        )
        ok = abs(az - target_az) < 0.01 and abs(el - target_el) < 0.01
        all_pass &= ok
        print(
            f"  {'PASS' if ok else 'FAIL'}: target=({target_az:.1f}°, {target_el:.1f}°) "
            f"recovered=({az:.4f}°, {el:.4f}°)"
        )

    print("-> 全ケース PASS\n" if all_pass else "-> 失敗あり\n")
    return all_pass


def test_equirect_to_perspective():
    print("=== テスト2: equirect_to_perspective スモークテスト ===")

    # 合成Equirectangular画像（BGR, 960×1920）
    np.random.seed(42)
    frame = np.random.randint(0, 256, (960, 1920, 3), dtype=np.uint8)

    patch = equirect_to_perspective(frame, yaw_deg=30.0, pitch_deg=10.0, fov_deg=90, out_size=448)

    assert patch.shape == (448, 448, 3), f"想定外のshape: {patch.shape}"
    assert patch.dtype == np.uint8, f"想定外のdtype: {patch.dtype}"

    # RGB確認（BGRのままだと赤と青が逆になるので統計的に判定は難しいが shape・dtype を確認）
    print(f"  patch.shape: {patch.shape}")
    print(f"  patch.dtype: {patch.dtype}")
    print(f"  patch.min/max: {patch.min()} / {patch.max()}")
    print("  PASS: shape・dtype 正常\n")
    return True


if __name__ == "__main__":
    ok1 = test_round_trip()
    ok2 = test_equirect_to_perspective()
    ok3 = test_patch_norm_to_spherical()

    if ok1 and ok2 and ok3:
        print("全テスト PASS")
        sys.exit(0)
    else:
        print("テスト失敗あり")
        sys.exit(1)
