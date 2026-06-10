import cv2
import numpy as np
import py360convert


def equirect_to_perspective(frame, yaw_deg, pitch_deg, fov_deg=90, out_size=448):
    """
    Equirectangularフレームから透視投影パッチを切り出す。

    Parameters
    ----------
    frame : np.ndarray
        BGR uint8 equirectangular画像 shape (H, W, 3)。THETA Xから取得したフレーム。
    yaw_deg : float
        水平方向（方位角）。正が右。範囲 [-180, 180]。
    pitch_deg : float
        垂直方向（仰角）。正が上。範囲 [-90, 90]。
    fov_deg : float
        画角（水平・垂直共通）。デフォルト 90°。
    out_size : int
        出力パッチの一辺（ピクセル）。Gaze-LLEの入力サイズ 448 がデフォルト。

    Returns
    -------
    np.ndarray
        RGB uint8 透視投影パッチ shape (out_size, out_size, 3)。
        Gaze-LLEの transform() にそのまま渡せる。
    """
    patch_bgr = py360convert.e2p(
        frame,
        fov_deg=fov_deg,
        u_deg=yaw_deg,
        v_deg=pitch_deg,
        out_hw=(out_size, out_size),
        mode='bilinear',
    )
    # BGR→RGB変換。cv2スライスは負ストライドになるため cvtColor を使う。
    return cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)


def patch_norm_to_spherical(col_norm, row_norm, yaw_deg, pitch_deg, fov_deg=90):
    """透視投影パッチ内の正規化座標 [0, 1] を 360°ワールド球面座標に変換する。

    equirect_to_perspective() で yaw_deg/pitch_deg/fov_deg を中心に切り出したパッチ上の
    任意の点（頭部 BBox 中心や視線ヒートマップのピークなど）を、ワールド空間の
    方位角・仰角へ写像する。heatmap_to_spherical の中核計算（Step2-4）と共有する。

    Parameters
    ----------
    col_norm : float
        パッチ内の水平位置 [0, 1]。0 が左端、1 が右端。
    row_norm : float
        パッチ内の垂直位置 [0, 1]。0 が上端、1 が下端。
    yaw_deg, pitch_deg : float
        equirect_to_perspective() に渡したカメラの方位角・仰角（度）。
    fov_deg : float
        equirect_to_perspective() に渡した画角（度）。

    Returns
    -------
    tuple[float, float]
        (azimuth_deg, elevation_deg)。azimuth ∈ [-180, 180]、elevation ∈ [-90, 90]。
    """
    # --- ステップ1: 正規化座標 → カメラフレーム内レイ方向 ---
    # py360convert の xyzpers は linspace(-tan_half, +tan_half, N) の格子を使う
    tan_half = np.tan(np.deg2rad(fov_deg / 2.0))
    x_cam = -tan_half + col_norm * 2.0 * tan_half
    # 画像行は下が正。カメラフレームのy軸は上が正なので符号を反転。
    y_cam = tan_half - row_norm * 2.0 * tan_half
    z_cam = 1.0

    norm = np.sqrt(x_cam ** 2 + y_cam ** 2 + z_cam ** 2)
    cam_ray = np.array([x_cam / norm, y_cam / norm, z_cam / norm])

    # --- ステップ2: カメラフレーム → ワールドフレームへ回転 ---
    # py360convert e2p.py の符号に合わせる:
    #   u = -deg2rad(u_deg)  (yaw)
    #   v =  deg2rad(v_deg)  (pitch)
    # world_ray = cam_ray @ Rx(v) @ Ry(u)
    u = -np.deg2rad(yaw_deg)
    v = np.deg2rad(pitch_deg)

    cos_u, sin_u = np.cos(u), np.sin(u)
    cos_v, sin_v = np.cos(v), np.sin(v)

    Rx = np.array([
        [1,     0,      0],
        [0,  cos_v, -sin_v],
        [0,  sin_v,  cos_v],
    ])
    Ry = np.array([
        [ cos_u, 0, sin_u],
        [     0, 1,     0],
        [-sin_u, 0, cos_u],
    ])

    world_ray = cam_ray @ Rx @ Ry

    # --- ステップ3: 直交座標 → 球面座標 ---
    azimuth_deg = np.degrees(np.arctan2(world_ray[0], world_ray[2]))
    elevation_deg = np.degrees(
        np.arctan2(world_ray[1], np.hypot(world_ray[0], world_ray[2]))
    )

    return float(azimuth_deg), float(elevation_deg)


def heatmap_to_spherical(heatmap, yaw_deg, pitch_deg, fov_deg=90):
    """
    Gaze-LLEの出力ヒートマップのピークを球面座標（方位角・仰角）に変換する。

    変換の流れ:
      soft-argmax でサブピクセルピーク算出
      → 透視投影の逆変換でカメラフレーム内レイ方向を計算
      → py360convert の e2p と同じ回転でワールドフレームに変換
      → 直交座標→球面座標

    Parameters
    ----------
    heatmap : np.ndarray
        float32 shape (64, 64)、値域 [0, 1]。Gaze-LLEの出力。
    yaw_deg : float
        equirect_to_perspective() に渡したカメラの水平方向（度）。
    pitch_deg : float
        equirect_to_perspective() に渡したカメラの垂直方向（度）。
    fov_deg : float
        equirect_to_perspective() に渡した画角（度）。

    Returns
    -------
    tuple[float, float]
        (azimuth_deg, elevation_deg) 360°ワールド空間上の球面座標（度）。
        azimuth ∈ [-180, 180]、elevation ∈ [-90, 90]。
    """
    H, W = heatmap.shape

    # --- Step 1: soft-argmax でサブピクセルピークを算出 ---
    total = float(heatmap.sum())
    if total < 1e-9:
        # ヒートマップが全ゼロ → カメラ中心方向をフォールバックとして返す
        return float(yaw_deg), float(pitch_deg)

    row_idx = np.arange(H, dtype=np.float64).reshape(H, 1)
    col_idx = np.arange(W, dtype=np.float64).reshape(1, W)
    peak_row = float((heatmap * row_idx).sum() / total)
    peak_col = float((heatmap * col_idx).sum() / total)

    # --- Step 2: パッチ正規化座標に直してワールド球面座標へ ---
    # peak_col/(W-1)・peak_row/(H-1) でパッチ内 [0,1] 正規化座標に変換し、共通ヘルパーに渡す。
    return patch_norm_to_spherical(
        peak_col / (W - 1), peak_row / (H - 1), yaw_deg, pitch_deg, fov_deg
    )
