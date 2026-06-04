import torch
from PIL import Image

# デバイス設定
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"使用デバイス: {device}")

# モデルロード
model, transform = torch.hub.load(
    'fkryan/gazelle',
    'gazelle_dinov2_vitb14_inout',
    pretrained=True,
    trust_repo=True
)
model = model.to(device).eval()
print("モデルロード成功")

# 推論
image = Image.open("test.jpg").convert("RGB")
inp = {
    "images": transform(image).unsqueeze(0).to(device),
    "bboxes": [[None]]
}

with torch.no_grad():
    out = model(inp)

heatmap = out["heatmap"][0][0].cpu().numpy()
inout = out["inout"][0][0].item()

print(f"ヒートマップのshape: {heatmap.shape}")
print(f"視線がフレーム内にある確率: {inout:.2f}")
print("動作確認成功！")
