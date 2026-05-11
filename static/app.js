/* global fetch, FormData, URL, document */

const $ = (id) => document.getElementById(id);

const THEME_KEY = "alexnet-viz-theme";

function getTheme() {
  const t = document.documentElement.getAttribute("data-theme");
  return t === "light" ? "light" : "dark";
}

function setTheme(theme) {
  const t = theme === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", t);
  try {
    localStorage.setItem(THEME_KEY, t);
  } catch (e) {
    /* ignore */
  }
  syncThemeToggle();
}

function syncThemeToggle() {
  const btn = $("theme-toggle");
  if (!btn) return;
  const dark = getTheme() === "dark";
  btn.textContent = dark ? "☀️ 白天模式" : "🌙 夜间模式";
  btn.setAttribute("aria-pressed", dark ? "true" : "false");
  btn.setAttribute(
    "title",
    dark ? "切换为浅色（白天）界面" : "切换为深色（夜间）界面"
  );
}

function toggleTheme() {
  setTheme(getTheme() === "dark" ? "light" : "dark");
}

function wireTheme() {
  const btn = $("theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => toggleTheme());
}

let lastFile = null;
let selected = { layer: null, channel: null };

async function refreshDevice() {
  const el = $("device-pill");
  try {
    const r = await fetch("/api/device");
    const d = await r.json();
    const dev = d.cuda_available ? `GPU：${d.cuda_device_name || "CUDA"}` : "CPU（未检测到 CUDA）";
    el.textContent = `计算设备：${d.device} · ${dev}`;
  } catch {
    el.textContent = "无法连接后端，请确认服务已启动";
  }
}

async function loadKernelsAndMeta() {
  const wrap = $("kernels-wrap");
  try {
    const r = await fetch("/api/layers");
    const d = await r.json();
    wrap.innerHTML = "";
    const img = document.createElement("img");
    img.alt = "Conv1 卷积核栅格";
    img.src = `data:image/png;base64,${d.kernels_grid_b64}`;
    wrap.appendChild(img);
  } catch (e) {
    wrap.innerHTML = `<p class="muted">加载卷积核失败：${String(e)}</p>`;
  }
}

function b64ToDataUrl(b64) {
  return `data:image/png;base64,${b64}`;
}

function setInputImages(inputB64) {
  const im = $("img-input");
  im.src = b64ToDataUrl(inputB64);
  im.classList.remove("hidden");
}

function clearOverlay() {
  const ov = $("img-overlay");
  ov.classList.add("hidden");
  ov.removeAttribute("src");
  $("heatmap-caption").textContent = "";
}

function renderTop5(top5) {
  const box = $("top5");
  if (!top5 || !top5.length) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");
  const items = top5.map((t, i) => `<li><strong>${(t.score * 100).toFixed(1)}%</strong> — ${t.label}</li>`).join("");
  box.innerHTML = `<div class="muted">ImageNet Top-5 预测</div><ol>${items}</ol>`;
}

function makeLayerCard(layer, index) {
  const card = document.createElement("div");
  card.className = "layer-card" + (index === 0 ? " open" : "");

  const head = document.createElement("div");
  head.className = "layer-head";
  head.innerHTML = `
    <div>
      <div class="layer-title">${friendlyLayerName(layer.name)}</div>
      <div class="layer-meta">${layer.h}×${layer.w} · ${layer.channels} 通道 · 展示 ${layer.tiles.length} 个</div>
    </div>
    <div class="chev">▼</div>
  `;

  const body = document.createElement("div");
  body.className = "layer-body";
  const grid = document.createElement("div");
  grid.className = "tile-grid";

  for (const t of layer.tiles) {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.dataset.layer = layer.name;
    tile.dataset.ch = String(t.ch);
    const im = document.createElement("img");
    im.src = b64ToDataUrl(t.img_b64);
    im.alt = `ch ${t.ch}`;
    const lab = document.createElement("span");
    lab.textContent = `通道 ${t.ch}`;
    tile.appendChild(im);
    tile.appendChild(lab);
    tile.addEventListener("click", (ev) => {
      ev.stopPropagation();
      onTileClick(layer.name, t.ch, tile);
    });
    grid.appendChild(tile);
  }

  body.appendChild(grid);
  card.appendChild(head);
  card.appendChild(body);

  head.addEventListener("click", () => {
    card.classList.toggle("open");
  });

  return card;
}

function friendlyLayerName(name) {
  const map = {
    "features.0": "Conv1（大卷积核 / 边缘与颜色）",
    "features.3": "Conv2",
    "features.6": "Conv3",
    "features.8": "Conv4",
    "features.10": "Conv5",
  };
  return map[name] || name;
}

async function onTileClick(layerName, channel, tileEl) {
  if (!lastFile) {
    alert("请先上传图片");
    return;
  }
  document.querySelectorAll(".tile.selected").forEach((n) => n.classList.remove("selected"));
  tileEl.classList.add("selected");
  selected = { layer: layerName, channel };

  $("heatmap-caption").textContent = "正在计算热力图…";
  const fd = new FormData();
  fd.append("file", lastFile, lastFile.name);
  fd.append("layer_name", layerName);
  fd.append("channel", String(channel));

  try {
    const r = await fetch("/api/heatmap", { method: "POST", body: fd });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(t || r.statusText);
    }
    const d = await r.json();
    const ov = $("img-overlay");
    ov.src = b64ToDataUrl(d.overlay_b64);
    ov.classList.remove("hidden");
    $("heatmap-caption").textContent = `层 ${layerName} · 通道 ${channel}：高亮区域对应该特征图在原图上的敏感位置（梯度×激活，上采样叠加）。`;
  } catch (e) {
    $("heatmap-caption").textContent = "";
    alert(`热力图请求失败：${e}`);
  }
}

async function runForward(file) {
  const root = $("layers-root");
  root.classList.add("loading");
  root.innerHTML = `<p class="muted">正在推理…（首次运行会下载预训练权重）</p>`;

  const fd = new FormData();
  fd.append("file", file, file.name);
  $("file-hint").textContent = `${file.name} · 处理中…`;

  try {
    const r = await fetch("/api/forward?max_ch=48", { method: "POST", body: fd });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(t || r.statusText);
    }
    const d = await r.json();
    lastFile = file;
    clearOverlay();
    setInputImages(d.input_b64);
    renderTop5(d.top5);

    root.innerHTML = "";
    d.layers.forEach((layer, i) => {
      root.appendChild(makeLayerCard(layer, i));
    });
    if (d.saved_path) {
      $("file-hint").textContent = `${file.name} · 已保存到 ${d.saved_path}`;
    } else {
      $("file-hint").textContent = file.name;
    }
  } catch (e) {
    $("file-hint").textContent = `${file.name} · 失败`;
    root.innerHTML = `<p class="muted">推理失败：${String(e)}</p>`;
  } finally {
    root.classList.remove("loading");
  }
}

function wireUpload() {
  $("file-input").addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    runForward(f);
  });
}

wireTheme();
syncThemeToggle();
refreshDevice();
loadKernelsAndMeta();
wireUpload();
