# gaze360

**RICOH THETA X（360度カメラ）+ Gaze-LLE によるリアルタイム視線推定システム**

360度映像に映る人物が「どこを見ているか」を推定し、視線先を球面座標（方位角・仰角）として出力します。

---

## 概要

- **入力**: RICOH THETA X の360度映像（Equirectangular形式）
- **処理**: 人物検出 → 透視投影変換 → 視線推定 → 球面座標への逆変換
- **出力**: 各人物の視線方向（方位角 `[-180°, 180°]`・仰角 `[-90°, 90°]`）と可視化映像
- **実行環境**: Mac（MPS）での開発、GPUサーバー（CUDA）でのリアルタイム推論

使用モデル：
- **[Gaze-LLE](https://github.com/fkryan/gazelle)**（CVPR 2025）— DINOv2エンコーダ + 軽量Transformerデコーダによる視線推定
- **[YOLOv8](https://github.com/ultralytics/ultralytics)** — 人物検出
- **[py360convert](https://github.com/sunset1995/py360convert)** — Equirectangular ↔ 透視投影変換
- **[theta-x-live](https://github.com/futosawaguchi/theta-x-live)** — THETA X 映像取得

---

## アーキテクチャ / データフロー

```
THETA X（Equirectangular 360度映像）
  ↓  ThetaStream
人物検出（YOLOv8）→ 各人物の (yaw, pitch)
  ↓  equirect_to_perspective
透視投影パッチ（448×448）
  ↓  GazeEstimator（Gaze-LLE）
視線ヒートマップ（64×64）
  ↓  heatmap_to_spherical
視線の球面座標（方位角・仰角）
  ↓
可視化（人物BBox・視線矢印・方位角/仰角テキスト）
```

---

## ディレクトリ構成

```
gaze360/
├── src/
│   ├── camera/theta_stream.py      # THETA X の MJPEG ストリーム取得
│   ├── detection/person_detector.py # YOLOv8 人物検出 → 球面座標
│   ├── gaze/estimator.py           # Gaze-LLE ラッパー（CUDA/MPS/CPU 自動選択）
│   ├── projection/equirect.py      # Equirectangular ↔ 透視投影 / 球面座標変換
│   └── pipeline.py                 # 統合パイプライン（メインエントリ）
├── scripts/
│   ├── test_projection.py          # 360度変換の数値テスト
│   ├── test_wrappers.py            # GazeEstimator / ThetaStream のテスト
│   ├── test_detector.py            # PersonDetector のテスト
│   └── relay_camera.py             # カメラ中継サーバー（同一LAN時の代替手段）
├── third_party/                    # サブモジュール（gazelle, theta-x-live）
├── run_gpu.sh.example              # GPUサーバー実行スクリプトのテンプレート
└── README.md
```

| モジュール | 主なクラス/関数 | 役割 |
|---|---|---|
| `camera/theta_stream.py` | `ThetaStream` | THETA X からフレームを取得（BGR numpy配列） |
| `detection/person_detector.py` | `PersonDetector`, `Detection` | 人物検出と球面座標への変換 |
| `gaze/estimator.py` | `GazeEstimator` | Gaze-LLE のロード・推論 |
| `projection/equirect.py` | `equirect_to_perspective`, `heatmap_to_spherical` | 投影変換と逆変換 |
| `pipeline.py` | `GazePipeline` | 全モジュールを統合 |

---

## セットアップ

### 共通

```bash
git clone --recurse-submodules https://github.com/futosawaguchi/gaze360.git
cd gaze360
python3 -m venv venv
source venv/bin/activate
```

### Mac（開発環境・MPS）

```bash
pip install torch torchvision
pip install -e third_party/gazelle
pip install -r third_party/theta-x-live/requirements.txt
pip install py360convert ultralytics timm
```

### GPUサーバー（本番環境・CUDA 12.1）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e third_party/gazelle
pip install -r third_party/theta-x-live/requirements.txt
pip install py360convert ultralytics timm
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121  # 高速化（任意）
```

### `.env` の作成

プロジェクトルート（`gaze360/`）に `.env` を作成します。

```
THETA_SERIAL=（カメラのシリアル番号）
THETA_IP=192.168.1.1            # CLモードでは実際のIP（例: 10.32.2.131）
THETA_PASSWORD=（省略可、デフォルトはシリアル番号）
```

> `.env` は `.gitignore` 済みです。`load_dotenv()` は実行ディレクトリの `.env` を読むため、プロジェクトルートに置いてください。

---

## 使い方

### ローカル（Mac）

cv2 ウィンドウに結果を表示します。

```bash
python -m src.pipeline
```

| キー | 動作 |
|---|---|
| `q` | 終了 |
| `s` | 現在のフレームを `outputs/` に保存 |

### 動画ファイルで確認

THETA X で録画した360度動画を入力にできます（デバッグ用）。

```bash
python -m src.pipeline --source path/to/video.mp4
```

### ブラウザ視聴（ストリームモード）

ディスプレイのない環境向け。MJPEG を HTTP 配信し、ブラウザで視聴します。

```bash
python -m src.pipeline --stream-port 8080
# ブラウザで http://localhost:8080 を開く（終了は Ctrl+C）
```

### GPUサーバーで実行

Mac から SSH トンネルで THETA X をGPUサーバーに中継し、GPUサーバー上で推論します。

```bash
cp run_gpu.sh.example run_gpu.sh
# run_gpu.sh 内の変数（ユーザー名・IP など）を自分の環境に合わせて編集
bash run_gpu.sh
# 表示された http://<サーバーIP>:<ポート> をブラウザで開く（終了は Ctrl+C）
```

`run_gpu.sh` は単一の SSH 接続で「THETA X へのトンネル」と「パイプライン起動」を同時に行います。SSH が切れるとカメラ入力も止まるため、`ServerAliveInterval` で接続を維持しています。

### オプション

| オプション | 説明 |
|---|---|
| `--source PATH` | 動画ファイルを入力に使う（省略時は THETA X カメラ） |
| `--stream-port PORT` | MJPEG を HTTP 配信（省略時は cv2 ウィンドウ表示） |
| `--scale FLOAT` | 表示・配信フレームの縮小倍率（省略時: ローカル `1.0` / ストリーム `0.5`） |

---

## テスト

```bash
python scripts/test_projection.py   # 360度変換のラウンドトリップ精度（誤差 < 0.01°）
python scripts/test_wrappers.py     # Gaze-LLE 推論・ThetaStream 設定の確認
python scripts/test_detector.py     # YOLOv8 人物検出と座標変換の確認
```

---

## 既知の制約・注意点

- **360度境界問題**：Equirectangular では左端と右端が連続していますが、平面に展開されているため、境界をまたぐ人物が2つの別人物として検出されます。**現在未対応（次フェーズで対応予定）**。
- **MPS は低速**：Mac（MPS）では DINOv2 のエンコードがボトルネックで約10FPS。リアルタイム性能が必要な場合は CUDA（GPUサーバー）を使用してください。
- **VPN 経由の配信はカクつく**：ストリームモードを VPN 越しに使うと帯域が不足しがちです。`--scale 0.35` などで配信フレームを縮小すると改善します（推論は原寸のまま行うため精度は変わりません）。
- **頭部 BBox は未指定**：1パッチに1人を切り出す運用のため、Gaze-LLE には `bbox=None` を渡しています（[Gaze-LLE 論文](https://arxiv.org/abs/2412.09586)が「1人のシーンでは BBox なしで有効」としているのに準拠）。

---

## 参考リンク

- Gaze-LLE 論文：https://arxiv.org/abs/2412.09586
- Gaze-LLE GitHub：https://github.com/fkryan/gazelle
- theta-x-live GitHub：https://github.com/futosawaguchi/theta-x-live
- py360convert：https://github.com/sunset1995/py360convert
- RICOH THETA X OSC API：https://api.ricoh/docs/theta-web-api-v2.1/
