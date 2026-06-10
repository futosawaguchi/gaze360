"""
gaze360 の外部出力契約。

`GazeResult` は「1人分の視線推定結果」を表すデータクラスで、gaze360 が外へ出す
公開データの形（contract）。可視化・ロボット制御など消費側はこれを受け取る。

互換性の方針:
    フィールドの **追加** は可。**削除・意味変更・型変更は不可**（後方互換を壊さない）。
    消費側（別リポジトリ等）は gaze360 を特定バージョンに pin して参照する想定。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GazeResult:
    """検出された1人分の視線推定結果。

    座標はすべて THETA カメラ座標系の球面座標（度）。
    方位角 yaw ∈ [-180, 180]（正が右）、仰角 pitch ∈ [-90, 90]（正が上）。
    """
    person_id: int            # フレーム内の人物番号（1始まり、confidence 降順）
    gaze_yaw: float           # 視線先の方位角。消費側の主役（どこを見ているか）
    gaze_pitch: float         # 視線先の仰角
    inout: float              # 視線がフレーム内にある確率 [0, 1]
    confidence: float         # 人物検出の信頼度 [0, 1]
    head_yaw: Optional[float] = None   # 頭部中心の方位角（頭部未検出なら None）
    head_pitch: Optional[float] = None  # 頭部中心の仰角（頭部未検出なら None）


def select_primary(results):
    """注視対象として追う「主たる人物」を1人選ぶ。

    最小実装は confidence 最大の人物（PersonDetector.detect() は既に降順ソート済み）。
    将来は「最大 bbox / 画面中心への近さ / 正面度」などのポリシーへ差し替え可能。

    Parameters
    ----------
    results : list[GazeResult]
        1フレーム分の視線推定結果。

    Returns
    -------
    GazeResult | None
        主たる人物。人物がいなければ None。
    """
    if not results:
        return None
    return max(results, key=lambda r: r.confidence)
