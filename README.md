# gaze360

**RICOH THETA X（360度カメラ）+ Gaze-LLE によるリアルタイム視線推定システム**

360度映像に映る人物が「どこを見ているか」を推定し、視線先を球面座標（方位角・仰角）の `GazeResult` として出力します。ロボット制御など外部アプリから視線データを利用できます。

---

## 概要

- **入力**: RICOH THETA X の360度映像（Equirectangular形式）
- **処理**: 人物検出 → 透視投影変換 → 頭部BBox検出 → 視線推定 → 球面座標への逆変換
- **出力**: 各人物の視線方向（方位角 `[-180°, 180°]`・仰角 `[-90°, 90°]`）を `GazeResult` として取得可能。可視化映像も生成
- **実行環境**: Mac（MPS）での開発、GPUサーバー（CUDA）でのリアルタイム推論

使用モデル：
- **[Gaze-LLE](https://github.com/fkryan/gazelle)**（CVPR 2025）— DINOv2エンコーダ + 軽量Transformerデコーダによる視線推定
- **[YOLOv8](https://github.com/ultralytics/ultralytics)** — 人物検出（YOLOv8）・頭部姿勢推定（YOLOv8-pose）
- **[py360convert](https://github.com/sunset1995/py360convert)** — Equirectangular ↔ 透視投影変換
- **[theta-x-live](https://github.com/futosawaguchi/theta-x-live)** — THETA X 映像取得

---

## アーキテクチャ / データフロー

```
THETA X（Equirectangular 360度映像）
  ↓  ThetaStream
人物検出（YOLOv8）→ 各人物の (yaw, pitch)   ※左右端マージで360度境界に対応
  ↓  equirect_to_perspective
透視投影パッチ（448×448）
  ↓  HeadDetector（YOLOv8-pose）→ 頭部BBox
  ↓  GazeEstimator（Gaze-LLE）
視線ヒートマップ（64×64）
  ↓  heatmap_to_spherical
視線の球面座標（方位角・仰角）
  ↓
GazeResult（外部出力） ＋ 可視化（人物BBox=緑 / 頭部BBox=マゼンタ / 視線矢印 / az・el）
```

---

## ディレクトリ構成

```
gaze360/
├── src/
│   ├── camera/theta_stream.py        # THETA X の MJPEG ストリーム取得
│   ├── detection/
│   │   ├── person_detector.py        # YOLOv8 人物検出 → 球面座標（360度境界マージ）
│   │   └── head_detector.py          # YOLOv8-pose 頭部BBox検出（パッチ内）
│   ├── gaze/
│   │   ├── estimator.py              # Gaze-LLE ラッパー（CUDA/MPS/CPU 自動選択）
│   │   └── result.py                 # GazeResult（外部出力契約）/ select_primary
│   ├── projection/equirect.py        # Equirectangular ↔ 透視投影 / 球面座標変換
│   └── pipeline.py                   # 統合パイプライン（メインエントリ）
├── scripts/
│   ├── test_projection.py            # 360度変換の数値テスト
│   ├── test_wrappers.py              # GazeEstimator / ThetaStream のテスト
│   ├── test_detector.py              # PersonDetector / HeadDetector のテスト
│   ├── test_result.py                # GazeResult / select_primary のテスト
│   └── relay_camera.py               # カメラ中継サーバー（同一LAN時の代替手段）
├── third_party/                      # サブモジュール（gazelle, theta-x-live）
├── run_gpu.sh.example                # GPUサーバー実行スクリプトのテンプレート
└── README.md
```

| モジュール | 主なクラス/関数 | 役割 |
|---|---|---|
| `camera/theta_stream.py` | `ThetaStream` | THETA X からフレームを取得（BGR numpy配列） |
| `detection/person_detector.py` | `PersonDetector`, `Detection` | 人物検出・球面座標変換・360度境界マージ |
| `detection/head_detector.py` | `HeadDetector` | パッチ内の頭部BBox検出（YOLOv8-pose） |
| `gaze/estimator.py` | `GazeEstimator` | Gaze-LLE のロード・推論 |
| `gaze/result.py` | `GazeResult`, `select_primary` | 視線推定結果の外部出力契約・主対象の選択 |
| `projection/equirect.py` | `equirect_to_perspective`, `heatmap_to_spherical`, `patch_norm_to_spherical` | 投影変換と逆変換 |
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
THETA_PASSWORD=デフォルトはシリアル番号
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
| `--stream-quality 1-100` | ストリーム配信時の JPEG 品質（デフォルト 55。VPN が細い場合は下げる） |
| `--scale FLOAT` | 表示・配信フレームの縮小倍率（省略時: ローカル `1.0` / ストリーム `0.5`） |

### プログラムから視線データを使う（外部連携）

`GazePipeline` に `on_results` コールバックを渡すと、毎フレームの `list[GazeResult]` を受け取れます（ロボット制御など外部アプリ向けの出口）。指定しなければ従来どおり可視化のみで挙動は変わりません。

```python
from src.pipeline import GazePipeline
from src.gaze.result import select_primary

def on_results(results):            # results: list[GazeResult]
    p = select_primary(results)     # 主たる人物（confidence 最大）
    if p and p.inout >= 0.3:
        print(p.person_id, p.gaze_yaw, p.gaze_pitch)  # 視線先の方位角・仰角

GazePipeline(source="path/to/video.mp4", on_results=on_results).run()
```

`GazeResult` の主なフィールド: `person_id, gaze_yaw, gaze_pitch, inout, confidence, head_yaw, head_pitch`
（座標は THETA カメラ座標系の球面座標・度。`head_*` は頭部未検出時 `None`）。

---

## テスト

```bash
python scripts/test_projection.py   # 360度変換のラウンドトリップ精度（誤差 < 0.01°）
python scripts/test_wrappers.py     # Gaze-LLE 推論・ThetaStream 設定の確認
python scripts/test_detector.py     # 人物検出・頭部BBox・360度境界マージの確認
python scripts/test_result.py       # GazeResult / select_primary の確認
```

---

## 既知の制約・注意点

- **360度境界**：Equirectangular の左右端をまたぐ人物は、左右端の断片を1人に統合（マージ）して扱います。視線矢印が境界をまたぐ場合は2分割で描画し、画面の横断を防ぎます。
- **MPS は低速**：Mac（MPS）では DINOv2 のエンコードがボトルネックで約1FPSしか出ません。リアルタイム性能が必要な場合は CUDA（GPUサーバー）を使用してください。
- **VPN 経由の配信はカクつく**：ストリームモードを VPN 越しに使うと帯域が不足しがちです。`--scale 0.35` で配信フレームを縮小、または `--stream-quality` を下げると改善します（推論は原寸のまま行うため精度は変わりません）。
- **頭部 BBox**：パッチ内で YOLOv8-pose により顔まわりの keypoint から頭部BBoxを構成し、Gaze-LLE に渡します（検出できない場合は `bbox=None` でフォールバック）。

---

## 参考リンク

- Gaze-LLE 論文：https://arxiv.org/abs/2412.09586
- Gaze-LLE GitHub：https://github.com/fkryan/gazelle
- theta-x-live GitHub：https://github.com/futosawaguchi/theta-x-live
- py360convert：https://github.com/sunset1995/py360convert
- RICOH THETA X OSC API：https://api.ricoh/docs/theta-web-api-v2.1/
