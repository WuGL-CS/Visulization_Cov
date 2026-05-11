# Visulization_Cov

基于 **PyTorch + torchvision 预训练 AlexNet** 的 Web 演示：分层卷积特征图浏览、**第一层卷积核可视化**，以及点击任意通道后在原图上叠加 **梯度×激活** 热力图（类激活 / Grad-CAM 思路），便于从底层纹理到高层部件理解 CNN 的分层表征。

**在线仓库：** [https://github.com/WuGL-CS/Visulization_Cov](https://github.com/WuGL-CS/Visulization_Cov)

---

## 功能概览

- 上传图像，按 ImageNet 预处理送入 AlexNet，展示 **5 个卷积层** 的特征图网格（默认每层前 48 个通道）。
- 点击某一通道，计算该通道对输入的敏感区域，**Jet 伪彩叠加**在原图上。
- **Conv1（11×11×3）** 真实权重以 8×8 栅格展示。
- 页面支持 **浅色 / 深色** 主题切换，偏好保存在浏览器 `localStorage`。
- 成功推理后，原始上传文件会保存到项目目录下的 **`uploads/`**（文件名带时间戳与随机后缀，见 `.gitignore`，默认不纳入版本库）。
- 推理设备 **自动选择 CUDA**，不可用时回退 **CPU**。

---

## 技术栈

| 组件 | 说明 |
|------|------|
| Python 3.10+ | 建议版本 |
| PyTorch / torchvision | `AlexNet_Weights.IMAGENET1K_V1` |
| FastAPI + Uvicorn | HTTP API 与静态页 |
| 原生 HTML / CSS / JS | 前端无构建步骤 |

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/WuGL-CS/Visulization_Cov.git
cd Visulization_Cov
```

或使用 SSH：

```bash
git clone git@github.com:WuGL-CS/Visulization_Cov.git
cd Visulization_Cov
```

### 2. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

首次运行会从 PyTorch 官方源下载 AlexNet 权重（约 233MB），需联网。

### 3. 启动服务

```bash
python3 -m uvicorn server:app --host 127.0.0.1 --port 8765
```

浏览器打开：**http://127.0.0.1:8765/**

局域网访问可将 `--host` 改为 `0.0.0.0`，并注意防火墙与数据安全。

---

## 主要 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 可视化主页 |
| `GET` | `/api/device` | 当前设备（`cuda` / `cpu`） |
| `GET` | `/api/layers` | 卷积层元数据 + Conv1 卷积核栅格（Base64 PNG） |
| `POST` | `/api/forward` | `multipart/form-data` 上传图片；返回特征图缩略图、Top-5、`saved_path` |
| `POST` | `/api/heatmap` | 表单：`file`、`layer_name`、`channel`，返回叠加图 Base64 |

查询参数 `max_ch`（如 `/api/forward?max_ch=48`）控制每层展示的通道数上限。

---

## 项目结构

```
Visulization_Cov/
├── server.py           # FastAPI 应用、模型、Grad×Activation、落盘逻辑
├── requirements.txt
├── static/
│   ├── index.html
│   ├── style.css       # 含 data-theme 浅色/深色变量
│   └── app.js
├── uploads/            # 用户上传保存目录（运行时自动创建，已 gitignore）
├── Task4CNN.ts         # 历史/其它场景中的 TensorFlow.js 演示片段（可选）
├── LICENSE             # Apache-2.0
└── README.md
```

---

## 许可证

本项目以 [Apache License 2.0](LICENSE) 发布。

---

## 备注

- 仓库名 `Visulization_Cov` 为既有命名；若需与论文或课程统一，可在 GitHub 仓库设置中调整展示名称，无需改本地目录名。
- 生产环境部署时请自行限制上传大小、鉴权与 `uploads` 磁盘占用。
