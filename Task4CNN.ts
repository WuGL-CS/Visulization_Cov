// src/scenes/Task4CNN.ts
import * as tf from '@tensorflow/tfjs';
import { IScene } from './BaseScene';

export class Task4CNN implements IScene {
    private container: HTMLDivElement | null = null;
    private topBar: HTMLDivElement | null = null;
    private pageShell: HTMLDivElement | null = null;
    private model: tf.Sequential | null = null;
    private convLayerNames: string[] = [];
    private kernelContainer: HTMLDivElement | null = null;
    private originalCanvas: HTMLCanvasElement | null = null;
    private heatmapCanvas: HTMLCanvasElement | null = null;
    private layersContainer: HTMLDivElement | null = null;
    private uploadInput: HTMLInputElement | null = null;
    private currentImageData: ImageData | null = null;
    private app: any = null;
    private readonly handleResize = () => this.syncLayoutOffsets();

    private savedFeatureMaps: Map<string, tf.Tensor<tf.Rank.R4>> = new Map();

    async init(app: any): Promise<void> {
        this.app = app;

        if (this.app?.renderer?.background) {
            this.app.renderer.background.color = 0xffffff;
        }
        document.body.style.backgroundColor = '#ffffff';

        this.createTopBar();

        // 纯白主容器
        this.container = document.createElement('div');
        this.container.style.position = 'absolute';
        this.container.style.left = '20px';
        this.container.style.right = '20px';
        this.container.style.bottom = '20px';
        this.container.style.backgroundColor = '#ffffff';
        this.container.style.borderRadius = '12px';
        this.container.style.padding = '20px';
        this.container.style.color = '#1e293b';
        this.container.style.fontFamily = 'system-ui, sans-serif';
        this.container.style.overflow = 'auto';
        this.container.style.zIndex = '20';
        this.container.style.boxShadow = 'none';
        document.body.appendChild(this.container);
        window.addEventListener('resize', this.handleResize);
        this.syncLayoutOffsets();

        this.pageShell = document.createElement('div');
        this.pageShell.style.maxWidth = '1200px';
        this.pageShell.style.margin = '0 auto';
        this.pageShell.style.display = 'flex';
        this.pageShell.style.flexDirection = 'column';
        this.pageShell.style.gap = '24px';
        this.container.appendChild(this.pageShell);

        // 标题区域（简洁）
        const titleSection = document.createElement('div');
        titleSection.style.borderBottom = '1px solid #e2e8f0';
        titleSection.style.paddingBottom = '12px';
        titleSection.style.marginBottom = '8px';
        this.pageShell.appendChild(titleSection);

        const title = document.createElement('h2');
        title.textContent = '🧠 CNN 特征可视化探索 (AlexNet 风格)';
        title.style.margin = '0 0 6px 0';
        title.style.fontSize = '26px';
        title.style.fontWeight = '600';
        title.style.color = '#0f172a';
        titleSection.appendChild(title);

        const subtitle = document.createElement('p');
        subtitle.textContent = '展示第一层卷积核与多层特征图响应；点击任意通道查看对应热力图叠加。';
        subtitle.style.margin = '0';
        subtitle.style.color = '#64748b';
        subtitle.style.fontSize = '14px';
        titleSection.appendChild(subtitle);

        await this.buildAlexNet();

        // 左右布局（紧凑）
        const layout = document.createElement('div');
        layout.style.display = 'grid';
        layout.style.gridTemplateColumns = 'minmax(280px, 0.9fr) minmax(460px, 1.2fr)';
        layout.style.gap = '24px';
        layout.style.alignItems = 'start';
        this.pageShell.appendChild(layout);

        // 左侧面板
        const leftPanel = document.createElement('div');
        leftPanel.style.position = 'sticky';
        leftPanel.style.top = '100px';
        leftPanel.style.backgroundColor = '#f8fafc';
        leftPanel.style.borderRadius = '16px';
        leftPanel.style.padding = '16px';
        leftPanel.style.border = '1px solid #e2e8f0';
        layout.appendChild(leftPanel);

        const imgTitle = document.createElement('h3');
        imgTitle.textContent = '输入图像 + 热力图叠加';
        imgTitle.style.margin = '0 0 12px 0';
        imgTitle.style.fontSize = '18px';
        imgTitle.style.fontWeight = '600';
        imgTitle.style.color = '#1e293b';
        leftPanel.appendChild(imgTitle);

        const canvasWrapper = document.createElement('div');
        canvasWrapper.style.position = 'relative';
        canvasWrapper.style.display = 'block';
        canvasWrapper.style.borderRadius = '12px';
        canvasWrapper.style.overflow = 'hidden';
        canvasWrapper.style.backgroundColor = '#ffffff';
        canvasWrapper.style.border = '1px solid #e2e8f0';
        leftPanel.appendChild(canvasWrapper);

        this.originalCanvas = document.createElement('canvas');
        this.originalCanvas.width = 224;
        this.originalCanvas.height = 224;
        this.originalCanvas.style.width = '100%';
        this.originalCanvas.style.height = 'auto';
        this.originalCanvas.style.backgroundColor = '#f1f5f9';
        canvasWrapper.appendChild(this.originalCanvas);

        this.heatmapCanvas = document.createElement('canvas');
        this.heatmapCanvas.width = 224;
        this.heatmapCanvas.height = 224;
        this.heatmapCanvas.style.position = 'absolute';
        this.heatmapCanvas.style.top = '0';
        this.heatmapCanvas.style.left = '0';
        this.heatmapCanvas.style.width = '100%';
        this.heatmapCanvas.style.height = '100%';
        this.heatmapCanvas.style.pointerEvents = 'none';
        canvasWrapper.appendChild(this.heatmapCanvas);

        const uploadBtn = document.createElement('button');
        uploadBtn.textContent = '📤 上传图片';
        uploadBtn.style.marginTop = '16px';
        uploadBtn.style.padding = '8px 16px';
        uploadBtn.style.backgroundColor = '#f1f5f9';
        uploadBtn.style.border = '1px solid #cbd5e1';
        uploadBtn.style.borderRadius = '30px';
        uploadBtn.style.color = '#1e293b';
        uploadBtn.style.cursor = 'pointer';
        uploadBtn.style.fontSize = '14px';
        uploadBtn.style.fontWeight = '500';
        uploadBtn.onclick = () => this.uploadInput?.click();
        leftPanel.appendChild(uploadBtn);

        this.uploadInput = document.createElement('input');
        this.uploadInput.type = 'file';
        this.uploadInput.accept = 'image/*';
        this.uploadInput.style.display = 'none';
        this.uploadInput.onchange = (e) => this.handleImageUpload(e);
        leftPanel.appendChild(this.uploadInput);

        // 右侧面板
        const rightPanel = document.createElement('div');
        rightPanel.style.backgroundColor = '#f8fafc';
        rightPanel.style.borderRadius = '16px';
        rightPanel.style.padding = '16px';
        rightPanel.style.border = '1px solid #e2e8f0';
        layout.appendChild(rightPanel);

        const layersTitle = document.createElement('h3');
        layersTitle.textContent = '卷积层特征图';
        layersTitle.style.margin = '0 0 12px 0';
        layersTitle.style.fontSize = '18px';
        layersTitle.style.fontWeight = '600';
        layersTitle.style.color = '#1e293b';
        rightPanel.appendChild(layersTitle);

        this.layersContainer = document.createElement('div');
        this.layersContainer.style.display = 'flex';
        this.layersContainer.style.flexDirection = 'column';
        this.layersContainer.style.gap = '16px';
        rightPanel.appendChild(this.layersContainer);

        // 卷积核可视化板块（置于特征图板块之后，仅局部使用深色背景）
        const kernelSection = document.createElement('div');
        kernelSection.style.border = '1px solid #0f172a';
        kernelSection.style.borderRadius = '18px';
        kernelSection.style.padding = '20px';
        kernelSection.style.background = 'linear-gradient(180deg, #0f172a 0%, #020617 100%)';
        kernelSection.style.boxShadow = '0 16px 40px rgba(15, 23, 42, 0.18)';
        this.pageShell.appendChild(kernelSection);

        this.kernelContainer = document.createElement('div');
        this.kernelContainer.style.display = 'flex';
        this.kernelContainer.style.flexDirection = 'column';
        this.kernelContainer.style.gap = '16px';
        kernelSection.appendChild(this.kernelContainer);

        await this.renderKernelVisualizations();

        this.drawPlaceholderImage();
    }

    private createTopBar() {
        const existingBar = document.getElementById('conv-top-bar');
        if (existingBar) existingBar.remove();

        const bar = document.createElement('div');
        bar.id = 'conv-top-bar';
        bar.style.position = 'absolute';
        bar.style.top = '0';
        bar.style.left = '0';
        bar.style.right = '0';
        bar.style.padding = '16px 24px 12px 24px';
        bar.style.backgroundColor = '#ffffff';
        bar.style.fontFamily = 'system-ui, sans-serif';
        bar.style.zIndex = '100';
        bar.style.boxShadow = '0 1px 3px rgba(0,0,0,0.05)';
        bar.style.display = 'flex';
        bar.style.flexDirection = 'column';
        bar.style.gap = '12px';

        const taskBar = document.createElement('div');
        taskBar.style.display = 'flex';
        taskBar.style.flexWrap = 'wrap';
        taskBar.style.gap = '10px';
        taskBar.style.marginBottom = '6px';

        const tasks = [
            { id: 'task1', name: '任务一：卷积演示' },
            { id: 'task2', name: '任务二：傅里叶变换' },
            { id: 'task3', name: '任务三：梯度下降演示' },
            { id: 'task4', name: '任务四：CNN特征可视化' },
            { id: 'task5', name: '任务五：自注意力机制' },
            { id: 'task6', name: '任务六：GAN训练博弈' },
            { id: 'task7', name: '任务七：扩散模型演示' },
            { id: 'task8', name: '任务八：目标检测对比' }
        ];

        tasks.forEach(task => {
            const btn = document.createElement('button');
            btn.textContent = task.name;
            // 统一样式，无高亮
            btn.style.padding = '6px 14px';
            btn.style.borderRadius = '30px';
            btn.style.border = '1px solid #cbd5e1';
            btn.style.backgroundColor = '#f8fafc';
            btn.style.color = '#1e293b';
            btn.style.fontSize = '13px';
            btn.style.fontWeight = '500';
            btn.style.cursor = 'pointer';
            btn.style.transition = 'all 0.2s';
            btn.onmouseenter = () => {
                btn.style.backgroundColor = '#e2e8f0';
                btn.style.borderColor = '#94a3b8';
            };
            btn.onmouseleave = () => {
                btn.style.backgroundColor = '#f8fafc';
                btn.style.borderColor = '#cbd5e1';
            };
            btn.onclick = () => {
                if ((window as any).loadScene) {
                    (window as any).loadScene(task.id);
                } else {
                    console.warn('loadScene 未定义');
                }
            };
            taskBar.appendChild(btn);
        });

        bar.appendChild(taskBar);
        document.body.appendChild(bar);
        this.topBar = bar;
    }

    private syncLayoutOffsets() {
        if (!this.container) return;
        const topOffset = (this.topBar?.offsetHeight ?? 0) + 16;
        this.container.style.top = `${topOffset}px`;
    }

    private async buildAlexNet() {
        this.model?.dispose();
        const model = tf.sequential();

        model.add(tf.layers.conv2d({
            inputShape: [224, 224, 3],
            filters: 64,
            kernelSize: 11,
            strides: 4,
            padding: 'same',
            activation: 'relu',
            name: 'conv1'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 3, strides: 2, name: 'pool1' }));

        model.add(tf.layers.conv2d({
            filters: 256,
            kernelSize: 5,
            strides: 1,
            padding: 'same',
            activation: 'relu',
            name: 'conv2'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 3, strides: 2, name: 'pool2' }));

        model.add(tf.layers.conv2d({
            filters: 384,
            kernelSize: 3,
            strides: 1,
            padding: 'same',
            activation: 'relu',
            name: 'conv3'
        }));

        model.add(tf.layers.conv2d({
            filters: 384,
            kernelSize: 3,
            strides: 1,
            padding: 'same',
            activation: 'relu',
            name: 'conv4'
        }));

        model.add(tf.layers.conv2d({
            filters: 256,
            kernelSize: 3,
            strides: 1,
            padding: 'same',
            activation: 'relu',
            name: 'conv5'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 3, strides: 2, name: 'pool5' }));

        model.add(tf.layers.flatten({ name: 'flatten' }));
        model.add(tf.layers.dense({ units: 4096, activation: 'relu', name: 'fc1' }));
        model.add(tf.layers.dense({ units: 4096, activation: 'relu', name: 'fc2' }));
        model.add(tf.layers.dense({ units: 1000, activation: 'softmax', name: 'classifier' }));

        this.model = model;
        this.convLayerNames = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5'];

        this.initializeAlexNetStyleConv1Weights();

        console.log('AlexNet 卷积层:', this.convLayerNames);
    }

    private initializeAlexNetStyleConv1Weights() {
        if (!this.model) return;

        const conv1 = this.model.getLayer('conv1');
        const kernelH = 11;
        const kernelW = 11;
        const inChannels = 3;
        const outChannels = 64;

        const kernelValues = this.createAlexNetStyleConv1KernelValues(kernelH, kernelW, inChannels, outChannels);
        const kernelTensor = tf.tensor4d(kernelValues, [kernelH, kernelW, inChannels, outChannels]);
        const biasTensor = tf.zeros([outChannels]);

        conv1.setWeights([kernelTensor, biasTensor]);

        kernelTensor.dispose();
        biasTensor.dispose();
    }

    private createAlexNetStyleConv1KernelValues(
        kernelH: number,
        kernelW: number,
        inChannels: number,
        outChannels: number
    ): Float32Array {
        const values = new Float32Array(kernelH * kernelW * inChannels * outChannels);

        const colorModes = [
            [1.00, 1.00, 1.00],
            [1.15, 1.15, 1.15],
            [1.20, -0.70, -0.70],
            [-0.70, 1.20, -0.70],
            [-0.70, -0.70, 1.20],
            [1.15, 0.20, -0.95],
            [-0.95, 0.20, 1.15],
            [0.95, -1.05, 0.20],
            [0.20, -1.05, 0.95],
            [1.05, -0.35, -0.95],
            [-0.95, -0.35, 1.05],
            [0.55, 1.05, -0.95],
            [-0.95, 1.05, 0.55],
            [1.05, -0.95, 0.55],
            [0.55, -0.95, 1.05],
            [1.00, 0.65, -0.75]
        ];

        const orientations = [
            0,
            Math.PI / 16,
            Math.PI / 12,
            Math.PI / 8,
            Math.PI / 6,
            Math.PI / 5,
            Math.PI / 4,
            Math.PI / 3,
            Math.PI / 2,
            (2 * Math.PI) / 3,
            (3 * Math.PI) / 4,
            (5 * Math.PI) / 6,
            Math.PI
        ];

        for (let f = 0; f < outChannels; f++) {
            const family = Math.floor(f / 8);
            const local = f % 8;
            const theta = orientations[(local * 2 + family) % orientations.length];
            const color = colorModes[(f * 3 + family) % colorModes.length];

            const freq = 1.15 + (local % 4) * 0.32 + (family % 2) * 0.12;
            const phase = (local % 4) * Math.PI / 4;
            const sigmaX = 0.52 - (family % 3) * 0.045;
            const sigmaY = 0.20 + (local % 4) * 0.035;

            const temp = new Float32Array(kernelH * kernelW * inChannels);
            let mean = 0;

            for (let y = 0; y < kernelH; y++) {
                for (let x = 0; x < kernelW; x++) {
                    const nx = (x - (kernelW - 1) / 2) / ((kernelW - 1) / 2);
                    const ny = (y - (kernelH - 1) / 2) / ((kernelH - 1) / 2);

                    const xr = nx * Math.cos(theta) + ny * Math.sin(theta);
                    const yr = -nx * Math.sin(theta) + ny * Math.cos(theta);

                    const gaussianLong = Math.exp(
                        -((xr * xr) / (2 * sigmaX * sigmaX) + (yr * yr) / (2 * sigmaY * sigmaY))
                    );

                    const gaussianRound = Math.exp(-((nx * nx + ny * ny) / 0.32));
                    const gaussianCenter = Math.exp(-((nx * nx + ny * ny) / 0.14));
                    const gaussianSurround = Math.exp(-((nx * nx + ny * ny) / 0.60));

                    let pattern = 0;

                    if (family === 0) {
                        // 亮暗方向边缘
                        pattern = gaussianLong * Math.cos(Math.PI * freq * xr + phase);
                    } else if (family === 1) {
                        // 更强的单边缘响应
                        pattern = gaussianLong * Math.tanh(4.2 * xr);
                    } else if (family === 2) {
                        // 细条纹 / 纹理滤波
                        pattern = gaussianLong * Math.sin(Math.PI * (freq + 0.6) * xr + phase);
                    } else if (family === 3) {
                        // 中心-周围滤波器，类似 blob
                        pattern = 1.45 * gaussianCenter - 0.82 * gaussianSurround;
                    } else if (family === 4) {
                        // 横纵组合，产生角点和交叉结构
                        const gA = Math.exp(-(xr * xr) / 0.060 - (yr * yr) / 0.42);
                        const gB = Math.exp(-(yr * yr) / 0.060 - (xr * xr) / 0.42);
                        pattern = gA - 0.85 * gB;
                    } else if (family === 5) {
                        // 对角局部结构
                        pattern = gaussianRound * Math.cos(Math.PI * (1.6 + local * 0.12) * (xr + yr) + phase);
                    } else if (family === 6) {
                        // 色彩渐变 + 方向边缘混合
                        pattern = gaussianLong * (0.75 * Math.cos(Math.PI * freq * xr + phase) + 0.55 * Math.tanh(3.0 * yr));
                    } else {
                        // 高对比方向纹理
                        pattern = gaussianLong * Math.cos(Math.PI * (freq + 0.9) * xr + phase) * Math.cos(Math.PI * 0.9 * yr);
                    }

                    const envelope = Math.exp(-0.12 * (nx * nx + ny * ny));

                    for (let c = 0; c < inChannels; c++) {
                        const colorScale = color[c];
                        const smallNaturalVariation =
                            1 +
                            0.035 * Math.sin((f + 1) * 0.73 + x * 0.61 + c * 1.37) +
                            0.025 * Math.cos((f + 1) * 0.41 + y * 0.53 + c * 1.91);

                        const v = pattern * envelope * colorScale * smallNaturalVariation;
                        const localIdx = (y * kernelW + x) * inChannels + c;
                        temp[localIdx] = v;
                        mean += v;
                    }
                }
            }

            mean /= temp.length;

            let maxAbs = 0;
            let std = 0;

            for (let i = 0; i < temp.length; i++) {
                temp[i] -= mean;
                std += temp[i] * temp[i];
            }

            std = Math.sqrt(std / temp.length);

            for (let i = 0; i < temp.length; i++) {
                const softened = Math.tanh(temp[i] / Math.max(std * 1.85, 1e-6));
                temp[i] = softened;
                maxAbs = Math.max(maxAbs, Math.abs(softened));
            }

            const gain = 0.20 / Math.max(maxAbs, 1e-6);

            for (let y = 0; y < kernelH; y++) {
                for (let x = 0; x < kernelW; x++) {
                    for (let c = 0; c < inChannels; c++) {
                        const localIdx = (y * kernelW + x) * inChannels + c;
                        const tensorIdx = (((y * kernelW + x) * inChannels + c) * outChannels) + f;
                        values[tensorIdx] = temp[localIdx] * gain;
                    }
                }
            }
        }

        return values;
    }

    private drawPlaceholderImage() {
        const ctx = this.originalCanvas?.getContext('2d');
        if (!ctx) return;
        ctx.fillStyle = '#f1f5f9';
        ctx.fillRect(0, 0, 224, 224);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '14px sans-serif';
        ctx.fillText('上传图片', 78, 112);
        this.heatmapCanvas?.getContext('2d')?.clearRect(0, 0, 224, 224);
    }

    private async handleImageUpload(e: Event) {
        const input = e.target as HTMLInputElement;
        if (!input.files?.length) return;
        const file = input.files[0];
        const img = new Image();
        img.onload = async () => {
            const ctx = this.originalCanvas?.getContext('2d');
            if (!ctx) return;
            ctx.drawImage(img, 0, 0, 224, 224);
            this.heatmapCanvas?.getContext('2d')?.clearRect(0, 0, 224, 224);
            const imageData = ctx.getImageData(0, 0, 224, 224);
            this.currentImageData = imageData;
            await this.runInferenceAndDisplayFeatures(imageData);
        };
        img.src = URL.createObjectURL(file);
    }

    private async runInferenceAndDisplayFeatures(imageData: ImageData) {
        if (!this.model || this.convLayerNames.length === 0) return;
        this.savedFeatureMaps.forEach(t => t.dispose());
        this.savedFeatureMaps.clear();

        const tensor = tf.browser.fromPixels(imageData).toFloat().div(255);
        const batched = tensor.expandDims(0);

        const outputs = this.convLayerNames.map(name => this.model!.getLayer(name).output);
        const flatOutputs = outputs.flat() as tf.SymbolicTensor[];
        const intermediateModel = tf.model({ inputs: this.model.inputs[0], outputs: flatOutputs });
        const predictions = intermediateModel.predict(batched) as tf.Tensor<tf.Rank.R4>[];

        for (let i = 0; i < this.convLayerNames.length; i++) {
            const cloned = tf.clone(predictions[i]);
            this.savedFeatureMaps.set(this.convLayerNames[i], cloned);
        }

        tensor.dispose();
        batched.dispose();
        predictions.forEach(t => t.dispose());
        intermediateModel.dispose();

        await this.renderFeatureMaps();
    }

    private async renderKernelVisualizations() {
        if (!this.model || !this.kernelContainer) return;
        this.kernelContainer.innerHTML = '';

        const conv1 = this.model.getLayer('conv1');
        const weights = conv1.getWeights();
        if (weights.length === 0) {
            const empty = document.createElement('div');
            empty.textContent = '卷积核尚未初始化';
            empty.style.color = '#cbd5e1';
            empty.style.fontSize = '14px';
            empty.style.textAlign = 'center';
            empty.style.padding = '16px 0';
            this.kernelContainer.appendChild(empty);
            return;
        }

        const kernel = weights[0] as tf.Tensor<tf.Rank.R4>;
        const [kernelH, kernelW, inChannels, outChannels] = kernel.shape;
        const displayCount = Math.min(64, outChannels);
        const kernelData = await kernel.data();

        const sectionHeader = document.createElement('div');
        sectionHeader.style.display = 'flex';
        sectionHeader.style.flexWrap = 'wrap';
        sectionHeader.style.justifyContent = 'space-between';
        sectionHeader.style.alignItems = 'center';
        sectionHeader.style.gap = '16px';
        this.kernelContainer.appendChild(sectionHeader);

        const titleBlock = document.createElement('div');
        titleBlock.style.flex = '1';
        sectionHeader.appendChild(titleBlock);

        const kernelTitle = document.createElement('div');
        kernelTitle.textContent = '卷积核可视化 (conv1)';
        kernelTitle.style.fontSize = '22px';
        kernelTitle.style.fontWeight = '600';
        kernelTitle.style.color = '#f8fafc';
        titleBlock.appendChild(kernelTitle);

        const meta = document.createElement('div');
        meta.textContent = `展示 conv1 全部 ${displayCount} 个滤波器 · 每个滤波器为 ${kernelH}×${kernelW}×RGB · 强化方向边缘、颜色对抗与中心响应`;
        meta.style.fontSize = '13px';
        meta.style.color = '#94a3b8';
        meta.style.marginTop = '6px';
        titleBlock.appendChild(meta);

        const badge = document.createElement('div');
        badge.textContent = `${kernelH} × ${kernelW}`;
        badge.style.padding = '6px 10px';
        badge.style.borderRadius = '999px';
        badge.style.border = '1px solid rgba(148, 163, 184, 0.35)';
        badge.style.color = '#e2e8f0';
        badge.style.fontSize = '12px';
        badge.style.fontWeight = '600';
        badge.style.backgroundColor = 'rgba(15, 23, 42, 0.42)';
        sectionHeader.appendChild(badge);

        const gridContainer = document.createElement('div');
        gridContainer.style.background = '#000000';
        gridContainer.style.border = '1px solid rgba(148, 163, 184, 0.2)';
        gridContainer.style.borderRadius = '14px';
        gridContainer.style.padding = '16px';
        gridContainer.style.overflowX = 'auto';
        this.kernelContainer.appendChild(gridContainer);

        const grid = document.createElement('div');
        grid.style.display = 'grid';
        grid.style.gridTemplateColumns = 'repeat(8, 96px)';
        grid.style.gridAutoRows = '96px';
        grid.style.gap = '10px';
        grid.style.justifyContent = 'center';
        grid.style.width = 'fit-content';
        grid.style.margin = '0 auto';
        gridContainer.appendChild(grid);

        for (let i = 0; i < displayCount; i++) {
            const tile = document.createElement('div');
            tile.style.width = '96px';
            tile.style.height = '96px';
            tile.style.background = '#000000';
            tile.style.border = '1px solid #1e293b';
            tile.style.borderRadius = '4px';
            tile.style.overflow = 'hidden';
            tile.style.boxSizing = 'border-box';
            tile.style.display = 'flex';
            tile.style.alignItems = 'center';
            tile.style.justifyContent = 'center';

            const canvas = document.createElement('canvas');
            canvas.width = 11;
            canvas.height = 11;
            canvas.style.width = '88px';
            canvas.style.height = '88px';
            canvas.style.display = 'block';
            canvas.style.imageRendering = 'pixelated';

            this.drawKernelToCanvas(canvas, kernelData, kernelH, kernelW, inChannels, outChannels, i);
            tile.appendChild(canvas);
            grid.appendChild(tile);
        }

        const note = document.createElement('div');
        note.textContent = '说明：外层为 8×8 的 64 个滤波器排列；每个小图内部是一个完整的 11×11 RGB 卷积核权重图，与参考文章中 AlexNet conv1 权重可视化逻辑一致。';
        note.style.fontSize = '12px';
        note.style.color = '#94a3b8';
        note.style.marginTop = '4px';
        note.style.textAlign = 'center';
        this.kernelContainer.appendChild(note);
    }

    private drawKernelToCanvas(
        canvas: HTMLCanvasElement,
        kernelData: ArrayLike<number>,
        kernelH: number,
        kernelW: number,
        inChannels: number,
        outChannels: number,
        filterIdx: number
    ) {
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = kernelW;
        canvas.height = kernelH;
        ctx.clearRect(0, 0, kernelW, kernelH);

        const values = new Float32Array(kernelH * kernelW * 3);
        let minVal = Infinity;
        let maxVal = -Infinity;

        for (let y = 0; y < kernelH; y++) {
            for (let x = 0; x < kernelW; x++) {
                const base = (y * kernelW + x) * 3;
                for (let c = 0; c < 3; c++) {
                    const sourceChannel = Math.min(c, inChannels - 1);
                    const tensorIdx = (((y * kernelW + x) * inChannels + sourceChannel) * outChannels) + filterIdx;
                    const value = kernelData[tensorIdx] ?? 0;
                    values[base + c] = value;
                    minVal = Math.min(minVal, value);
                    maxVal = Math.max(maxVal, value);
                }
            }
        }

        const range = Math.max(maxVal - minVal, 1e-6);
        const imgData = ctx.createImageData(kernelW, kernelH);

        for (let y = 0; y < kernelH; y++) {
            for (let x = 0; x < kernelW; x++) {
                const base = (y * kernelW + x) * 3;

                let r = (values[base] - minVal) / range;
                let g = (values[base + 1] - minVal) / range;
                let b = (values[base + 2] - minVal) / range;

                const adjusted = this.applyAlexNetKernelDisplayEnhancement(r, g, b);
                r = adjusted.r;
                g = adjusted.g;
                b = adjusted.b;

                const idx = (y * kernelW + x) * 4;
                imgData.data[idx] = Math.round(r * 255);
                imgData.data[idx + 1] = Math.round(g * 255);
                imgData.data[idx + 2] = Math.round(b * 255);
                imgData.data[idx + 3] = 255;
            }
        }

        ctx.imageSmoothingEnabled = false;
        ctx.putImageData(imgData, 0, 0);
    }

    private applyAlexNetKernelDisplayEnhancement(
        r: number,
        g: number,
        b: number
    ): { r: number, g: number, b: number } {
        // 模拟参考文章中 “归一化 + 增强对比度/亮度/饱和度” 的显示效果
        const contrast = 2.25;
        const brightness = 0.03;
        const saturation = 1.65;
        const gamma = 0.78;

        let rr = this.clamp01((r - 0.5) * contrast + 0.5 + brightness);
        let gg = this.clamp01((g - 0.5) * contrast + 0.5 + brightness);
        let bb = this.clamp01((b - 0.5) * contrast + 0.5 + brightness);

        const gray = rr * 0.299 + gg * 0.587 + bb * 0.114;
        rr = this.clamp01(gray + (rr - gray) * saturation);
        gg = this.clamp01(gray + (gg - gray) * saturation);
        bb = this.clamp01(gray + (bb - gray) * saturation);

        rr = Math.pow(rr, gamma);
        gg = Math.pow(gg, gamma);
        bb = Math.pow(bb, gamma);

        return {
            r: this.clamp01(rr),
            g: this.clamp01(gg),
            b: this.clamp01(bb)
        };
    }

    private clamp01(value: number): number {
        return Math.max(0, Math.min(1, value));
    }

    private async renderFeatureMaps() {
        if (!this.layersContainer) return;
        this.layersContainer.innerHTML = '';

        for (const [layerName, tensor] of this.savedFeatureMaps.entries()) {
            const shape = tensor.shape;
            if (shape.length !== 4) continue;
            const height = shape[1];
            const width = shape[2];
            const channels = shape[3];
            const maxChannels = Math.min(16, channels);

            const panel = document.createElement('div');
            panel.style.backgroundColor = '#ffffff';
            panel.style.borderRadius = '12px';
            panel.style.overflow = 'hidden';
            panel.style.border = '1px solid #e2e8f0';
            panel.style.marginBottom = '4px';

            const header = document.createElement('div');
            header.style.padding = '12px 16px';
            header.style.backgroundColor = '#f8fafc';
            header.style.cursor = 'pointer';
            header.style.fontWeight = '600';
            header.style.color = '#1e293b';
            header.style.display = 'flex';
            header.style.justifyContent = 'space-between';
            header.style.alignItems = 'center';
            header.innerHTML = `
                <div>
                    <div style="font-weight: 700;">${layerName}</div>
                    <div style="font-size: 12px; color: #64748b;">${height}×${width} · ${channels}通道 · 展示前${maxChannels}个</div>
                </div>
                <div style="font-size: 12px;">▼</div>
            `;
            header.onclick = () => {
                const gridDiv = panel.querySelector('.feature-grid') as HTMLElement;
                if (gridDiv) gridDiv.style.display = gridDiv.style.display === 'none' ? 'flex' : 'none';
            };
            panel.appendChild(header);

            const grid = document.createElement('div');
            grid.className = 'feature-grid';
            grid.style.display = 'flex';
            grid.style.flexWrap = 'wrap';
            grid.style.gap = '8px';
            grid.style.padding = '12px';
            grid.style.maxHeight = '320px';
            grid.style.overflowY = 'auto';
            grid.style.backgroundColor = '#ffffff';
            panel.appendChild(grid);

            const slice = tensor.slice([0, 0, 0, 0], [1, height, width, maxChannels]);
            const data = await slice.data();
            slice.dispose();

            for (let c = 0; c < maxChannels; c++) {
                const tile = document.createElement('div');
                tile.style.display = 'flex';
                tile.style.flexDirection = 'column';
                tile.style.alignItems = 'center';
                tile.style.gap = '6px';
                tile.style.padding = '6px';
                tile.style.borderRadius = '10px';
                tile.style.backgroundColor = '#fafcff';
                tile.style.border = '1px solid #e2e8f0';
                tile.style.cursor = 'pointer';

                const canvas = document.createElement('canvas');
                canvas.width = 56;
                canvas.height = 56;
                canvas.style.width = '56px';
                canvas.style.height = '56px';
                canvas.style.borderRadius = '6px';
                canvas.style.border = '1px solid #cbd5e1';

                const ctx = canvas.getContext('2d')!;
                const imgData = ctx.createImageData(56, 56);
                const scaleH = height / 56;
                const scaleW = width / 56;
                let minVal = Infinity, maxVal = -Infinity;
                const channelVals = new Array(height * width);
                for (let i = 0; i < height * width; i++) {
                    const val = data[i * maxChannels + c];
                    channelVals[i] = val;
                    if (val < minVal) minVal = val;
                    if (val > maxVal) maxVal = val;
                }
                const range = maxVal - minVal;
                for (let y = 0; y < 56; y++) {
                    for (let x = 0; x < 56; x++) {
                        const srcY = Math.floor(y * scaleH);
                        const srcX = Math.floor(x * scaleW);
                        let val = channelVals[srcY * width + srcX];
                        let norm = range > 0 ? (val - minVal) / range : 0;
                        const gray = Math.floor(norm * 255);
                        const idx = (y * 56 + x) * 4;
                        imgData.data[idx] = gray;
                        imgData.data[idx + 1] = gray;
                        imgData.data[idx + 2] = gray;
                        imgData.data[idx + 3] = 255;
                    }
                }
                ctx.putImageData(imgData, 0, 0);

                canvas.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.generateHeatmapFromFeature(layerName, c, height, width);
                });

                const label = document.createElement('div');
                label.textContent = `ch ${c}`;
                label.style.fontSize = '10px';
                label.style.fontWeight = '500';
                label.style.color = '#475569';

                tile.appendChild(canvas);
                tile.appendChild(label);
                grid.appendChild(tile);
            }
            this.layersContainer.appendChild(panel);
        }
    }

    private async generateHeatmapFromFeature(layerName: string, channelIdx: number, featH: number, featW: number) {
        if (!this.currentImageData) {
            alert('请先上传图片');
            return;
        }
        const fullTensor = this.savedFeatureMaps.get(layerName);
        if (!fullTensor) {
            console.error('特征图张量不存在');
            return;
        }

        const channelTensor = fullTensor.slice([0, 0, 0, channelIdx], [1, featH, featW, 1]);
        const data = await channelTensor.data();
        channelTensor.dispose();

        const heatmapData = new Float32Array(224 * 224);
        const scaleH = featH / 224;
        const scaleW = featW / 224;
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < data.length; i++) {
            if (data[i] < min) min = data[i];
            if (data[i] > max) max = data[i];
        }
        const range = max - min;
        for (let y = 0; y < 224; y++) {
            for (let x = 0; x < 224; x++) {
                const srcY = Math.floor(y * scaleH);
                const srcX = Math.floor(x * scaleW);
                let val = data[srcY * featW + srcX];
                let norm = range > 0 ? (val - min) / range : 0;
                heatmapData[y * 224 + x] = norm;
            }
        }

        const ctx = this.heatmapCanvas?.getContext('2d');
        const imgCtx = this.originalCanvas?.getContext('2d');
        if (!ctx || !imgCtx) return;
        const originalImageData = imgCtx.getImageData(0, 0, 224, 224);
        ctx.putImageData(originalImageData, 0, 0);
        for (let y = 0; y < 224; y++) {
            for (let x = 0; x < 224; x++) {
                const intensity = heatmapData[y * 224 + x];
                if (intensity < 0.2) continue;
                const color = this.jetColormap(intensity);
                ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.6)`;
                ctx.fillRect(x, y, 1, 1);
            }
        }
    }

    private jetColormap(t: number): { r: number, g: number, b: number } {
        const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)));
        const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)));
        const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)));
        return { r: Math.floor(r * 255), g: Math.floor(g * 255), b: Math.floor(b * 255) };
    }

    destroy(): void {
        window.removeEventListener('resize', this.handleResize);
        this.savedFeatureMaps.forEach(t => t.dispose());
        this.savedFeatureMaps.clear();
        this.model?.dispose();
        this.topBar?.remove();
        this.container?.remove();
        this.uploadInput?.remove();
    }
}