import numpy as np
import torch
from PIL import Image


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GazeEstimator:
    """Gaze-LLE モデルのロードと推論をまとめたラッパー。

    Usage:
        estimator = GazeEstimator()
        heatmap, inout = estimator.predict(rgb_patch)
        # または複数人
        results = estimator.predict_batch(rgb_patches, bboxes_list)
    """

    MODEL_NAME = "gazelle_dinov2_vitb14_inout"

    def __init__(self, device=None):
        """
        Parameters
        ----------
        device : torch.device | None
            推論デバイス。None の場合は CUDA > MPS > CPU の順で自動選択。
        """
        self.device = device or _get_device()
        print(f"[GazeEstimator] デバイス: {self.device}")

        self.model, self.transform = torch.hub.load(
            "fkryan/gazelle",
            self.MODEL_NAME,
            pretrained=True,
            trust_repo=True,
        )
        self.model = self.model.to(self.device).eval()
        print("[GazeEstimator] モデルロード完了")

    def predict(self, rgb_patch, bbox=None):
        """1人分の視線を推定する。

        Parameters
        ----------
        rgb_patch : np.ndarray
            RGB uint8 透視投影パッチ shape (H, W, 3)。
            equirect_to_perspective() の出力をそのまま渡せる。
        bbox : tuple | None
            頭部バウンディングボックス (xmin, ymin, xmax, ymax) の正規化座標 [0, 1]。
            None を渡すとモデルが画像全体から自動推定する。

        Returns
        -------
        heatmap : np.ndarray
            float32 shape (64, 64)、値域 [0, 1]。視線先の確率マップ。
        inout : float
            視線がフレーム内にある確率 [0, 1]。1 に近いほどフレーム内を見ている。
        """
        results = self._forward([rgb_patch], [[bbox]])
        heatmap = results["heatmap"][0][0].cpu().numpy()
        inout = results["inout"][0][0].item()
        return heatmap, inout

    def predict_multi(self, rgb_patch, bboxes):
        """1つのパッチ内に複数人いる場合の視線を一括推定する。

        Gaze-LLE はシーンエンコードを1回だけ行うため、
        複数人を個別に推定するより大幅に効率的。

        Parameters
        ----------
        rgb_patch : np.ndarray
            RGB uint8 透視投影パッチ shape (H, W, 3)。
        bboxes : list[tuple | None]
            人数分のバウンディングボックスリスト。各要素は (xmin, ymin, xmax, ymax) または None。

        Returns
        -------
        heatmaps : list[np.ndarray]
            人数分の heatmap リスト。各要素 shape (64, 64)。
        inouts : list[float]
            人数分の inout スコアリスト。
        """
        results = self._forward([rgb_patch], [bboxes])
        n = len(bboxes)
        heatmaps = [results["heatmap"][0][i].cpu().numpy() for i in range(n)]
        inouts = [results["inout"][0][i].item() for i in range(n)]
        return heatmaps, inouts

    def _forward(self, rgb_patches, bboxes_list):
        """内部推論メソッド。PIL変換・バッチ化・デバイス転送を処理する。"""
        tensors = []
        for patch in rgb_patches:
            pil = Image.fromarray(patch)
            tensors.append(self.transform(pil))
        images = torch.stack(tensors).to(self.device)

        inp = {"images": images, "bboxes": bboxes_list}
        with torch.no_grad():
            return self.model(inp)
