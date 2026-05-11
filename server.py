"""
AlexNet 分层可视化 + 通道级 Grad×Activation 热力图 + Conv1 卷积核可视化。
自动使用 CUDA（若可用），否则 CPU。
"""
from __future__ import annotations

import base64
import io
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from torchvision import models
from torchvision.models import AlexNet_Weights

# -----------------------------------------------------------------------------
# Device & model (singleton)
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[AlexNet Viz] Using device: {DEVICE}")

WEIGHTS = AlexNet_Weights.IMAGENET1K_V1
MODEL = models.alexnet(weights=WEIGHTS).to(DEVICE).eval()
PREPROCESS = WEIGHTS.transforms()


def _collect_conv2d_modules() -> list[tuple[str, torch.nn.Module]]:
    out: list[tuple[str, torch.nn.Module]] = []
    for name, m in MODEL.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            out.append((name, m))
    return out


CONV_LAYERS = _collect_conv2d_modules()


def _tensor_to_png_b64(img_chw: torch.Tensor) -> str:
    """img_chw: float [C,H,W] in [0,1] or normalized — convert to uint8 RGB PNG base64."""
    x = img_chw.detach().cpu().clamp(0, 1)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    arr = (x.numpy() * 255.0).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _normalize01(t: torch.Tensor) -> torch.Tensor:
    t = t - t.min()
    m = t.max().clamp_min(1e-8)
    return t / m


def _jet_on_gray(cam_hw: torch.Tensor) -> torch.Tensor:
    """cam_hw in [0,1], returns RGB [3,H,W] float [0,1]."""
    t = cam_hw.clamp(0, 1)
    r = (1.5 - (4 * t - 3).abs()).clamp(0, 1)
    g = (1.5 - (4 * t - 2).abs()).clamp(0, 1)
    b = (1.5 - (4 * t - 1).abs()).clamp(0, 1)
    return torch.stack([r, g, b], dim=0)


def _overlay_heatmap(
    image_rgb01: torch.Tensor, heat_hw: torch.Tensor, alpha: float = 0.55
) -> torch.Tensor:
    """image_rgb01 [3,224,224], heat_hw [224,224] in [0,1]."""
    jet = _jet_on_gray(heat_hw)
    return (1 - alpha) * image_rgb01 + alpha * jet


def preprocess_pil(pil: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        batch_net: [1,3,224,224] for model (normalized)
        rgb01: [3,224,224] for display overlay (ImageNet norm undone approximately via denorm)
    """
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    t = PREPROCESS(pil).unsqueeze(0).to(DEVICE)
    # Approximate inverse for overlay: use same tensor scaled to 0-1 for visualization
    rgb = t[0].clone().cpu()
    # Standard ImageNet denorm for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb01 = (rgb * std + mean).clamp(0, 1)
    return t, rgb01.to(DEVICE)


def forward_save_activations(x: torch.Tensor) -> dict[str, torch.Tensor]:
    acts: dict[str, torch.Tensor] = {}
    hooks: list[Any] = []

    def make_hook(name: str):
        def hook(_m, _inp, out):
            acts[name] = out.detach()

        return hook

    for name, module in CONV_LAYERS:
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = MODEL(x)

    for h in hooks:
        h.remove()

    return acts


def channel_grad_activation_map(
    x: torch.Tensor, layer_name: str, channel: int
) -> torch.Tensor:
    """
    通道级类激活风格图：对选定卷积层通道的均值激活求 ∂/∂A，
    使用 ReLU(梯度 × 激活) 得到空间敏感图，再上采样到输入分辨率。
    与 Grad-CAM 思想一致（该通道对自身的标量目标求导后加权特征图）。
    """
    if channel < 0:
        raise ValueError("channel must be non-negative")

    module = dict(MODEL.named_modules()).get(layer_name)
    if module is None or not isinstance(module, torch.nn.Conv2d):
        raise ValueError(f"Unknown conv layer: {layer_name}")

    x = x.clone().detach().requires_grad_(True)
    activation: torch.Tensor | None = None

    def fwd_hook(_m, _inp, out):
        nonlocal activation
        activation = out

    h = module.register_forward_hook(fwd_hook)
    try:
        _ = MODEL(x)
        if activation is None:
            raise RuntimeError("activation hook failed")
        act = activation
        if channel >= act.shape[1]:
            raise ValueError(
                f"channel {channel} out of range (0..{act.shape[1] - 1})"
            )
        loss = act[0, channel].mean()
        (grad,) = torch.autograd.grad(
            loss,
            act,
            retain_graph=False,
            create_graph=False,
        )
        cam = F.relu(grad[0, channel] * act[0, channel].detach())
        cam = _normalize01(cam)
        cam_up = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return cam_up.clamp(0, 1)
    finally:
        h.remove()


def conv1_kernels_png_grid() -> str:
    """Single PNG grid of 8x8 conv1 filters, RGB."""
    w = MODEL.features[0].weight.data.clone()  # [64,3,11,11]
    w = w - w.amin(dim=(1, 2, 3), keepdim=True)
    w = w / (w.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6))

    grid_h, grid_w = 8, 8
    kh, kw = 11, 11
    pad = 2
    canvas = torch.zeros(3, grid_h * (kh + pad) - pad, grid_w * (kw + pad) - pad)
    for i in range(64):
        r, c = i // grid_w, i % grid_w
        y0 = r * (kh + pad)
        x0 = c * (kw + pad)
        canvas[:, y0 : y0 + kh, x0 : x0 + kw] = w[i]
    return _tensor_to_png_b64(canvas)


def build_layer_tiles(
    acts: dict[str, torch.Tensor],
    layer_name: str,
    max_channels: int = 48,
    thumb: int = 64,
) -> dict[str, Any]:
    t = acts[layer_name][0]  # [C,H,W]
    c_full, h, w = t.shape
    n_show = min(max_channels, c_full)
    tiles: list[dict[str, Any]] = []
    for ci in range(n_show):
        sl = t[ci : ci + 1]
        sl = _normalize01(sl)
        up = F.interpolate(
            sl.unsqueeze(0), size=(thumb, thumb), mode="bilinear", align_corners=False
        )[0]
        up3 = up.repeat(3, 1, 1)
        tiles.append({"ch": ci, "img_b64": _tensor_to_png_b64(up3)})
    return {
        "name": layer_name,
        "channels": c_full,
        "h": h,
        "w": w,
        "tiles": tiles,
    }


# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(title="AlexNet CNN Visualization")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 用户上传图片落盘目录（相对项目根目录）
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"


def _safe_upload_filename(original: str | None) -> str:
    if not original or not str(original).strip():
        return "image"
    base = Path(str(original)).name
    base = re.sub(r"[^\w.\-]+", "_", base, flags=re.UNICODE).strip("._") or "image"
    if len(base) > 120:
        stem = Path(base).stem[:80]
        suf = Path(base).suffix[:20]
        base = stem + suf
    return base


def save_user_upload_bytes(raw: bytes, original_filename: str | None) -> str:
    """
    将用户上传的原始字节写入 uploads/，返回相对项目根的路径（POSIX 风格字符串）。
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    safe = _safe_upload_filename(original_filename)
    out_path = UPLOAD_DIR / f"{stamp}_{short}_{safe}"
    out_path.write_bytes(raw)
    rel = out_path.relative_to(Path(__file__).resolve().parent)
    return rel.as_posix()


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Missing static/index.html</h1>", status_code=500)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/device")
def api_device():
    return {
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
    }


@app.get("/api/layers")
def api_layers():
    layers = []
    for name, m in CONV_LAYERS:
        layers.append({"name": name, "out_channels": m.out_channels})
    return {"layers": layers, "kernels_grid_b64": conv1_kernels_png_grid()}


@app.post("/api/forward")
async def api_forward(file: UploadFile = File(...), max_ch: int = 48):
    try:
        raw = await file.read()
        pil = Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    try:
        saved_path = save_user_upload_bytes(raw, file.filename)
    except OSError as e:
        raise HTTPException(500, f"无法保存上传文件: {e}") from e

    x, rgb01 = preprocess_pil(pil)
    acts = forward_save_activations(x)
    layers_payload = []
    for name, _ in CONV_LAYERS:
        layers_payload.append(build_layer_tiles(acts, name, max_channels=max_ch))

    input_b64 = _tensor_to_png_b64(rgb01)

    # top-1 class (optional, for pedagogy)
    with torch.no_grad():
        logits = MODEL(x)
        prob = F.softmax(logits[0], dim=0)
        topv, topi = prob.topk(5)
    idx_to_name = WEIGHTS.meta["categories"]
    topk = [
        {"index": int(i), "score": float(v), "label": idx_to_name[int(i)]}
        for v, i in zip(topv.tolist(), topi.tolist())
    ]

    return JSONResponse(
        {
            "input_b64": input_b64,
            "layers": layers_payload,
            "top5": topk,
            "saved_path": saved_path,
        }
    )


@app.post("/api/heatmap")
async def api_heatmap(
    file: UploadFile = File(...),
    layer_name: str = Form(...),
    channel: int = Form(...),
):
    try:
        raw = await file.read()
        pil = Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    x, rgb01 = preprocess_pil(pil)
    try:
        heat = channel_grad_activation_map(x, layer_name, channel)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e)) from e

    overlay = _overlay_heatmap(rgb01, heat, alpha=0.52)
    return JSONResponse(
        {
            "overlay_b64": _tensor_to_png_b64(overlay),
            "heat_b64": _tensor_to_png_b64(_jet_on_gray(heat)),
        }
    )


class HeatmapJsonBody(BaseModel):
    image_b64: str
    layer_name: str
    channel: int


@app.post("/api/heatmap_json")
def api_heatmap_json(body: HeatmapJsonBody):
    try:
        raw = base64.standard_b64decode(body.image_b64)
        pil = Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Invalid base64 image: {e}") from e

    x, rgb01 = preprocess_pil(pil)
    try:
        heat = channel_grad_activation_map(x, body.layer_name, body.channel)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e)) from e

    overlay = _overlay_heatmap(rgb01, heat, alpha=0.52)
    return {
        "overlay_b64": _tensor_to_png_b64(overlay),
        "heat_b64": _tensor_to_png_b64(_jet_on_gray(heat)),
    }
