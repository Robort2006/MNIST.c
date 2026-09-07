# MNIST 手写数字识别 —— 纯 C 实现推理

Python 训练 + 纯 C 部署的手写数字识别项目。神经网络推理完全用 C 语言手写，不依赖任何深度学习框架，可直接移植到嵌入式平台（如 STM32）。

## 项目结构

```
MNIST.c/
├── c/
│   ├── matrix.h/.c      # 底层运算：矩阵乘+偏置、ReLU、Softmax、Argmax
│   ├── infer.h/.c       # 推理接口：前向传播
│   ├── preprocess.h/.c  # 图像预处理：裁剪+缩放+居中（对齐MNIST格式）
│   ├── bmp.h/.c         # BMP图片解析（24位，28x28）
│   ├── main.c            # 主程序：批量测试 / 单张BMP识别
│   └── weights.h         # 训练导出的权重（101770个float参数）
├── python/
│   ├── train.py          # 训练模型（含数据增强）+ 导出weights.h
│   └── export_bmp.py     # 从MNIST测试集导出BMP测试图片
├── download_mnist.py     # MNIST数据集下载工具
└── README.md
```

## 环境要求

- Python 3.10+（训练用，需 PyTorch CPU 版）
- GCC / MinGW（编译 C 代码）
- 约 100MB 磁盘空间（含数据集）

## 快速开始

### 1. 下载数据集

```bash
py download_mnist.py
```

数据集会下载到 `data/MNIST/raw/`。

### 2. 训练模型并导出权重

```bash
py python/train.py
```

训练完成后权重自动导出到 `c/weights.h`。

> 训练使用数据增强（随机平移±3px、旋转±10°、缩放0.9~1.1），提升模型对位置/角度/大小的鲁棒性。

### 3. 编译 C 推理程序

```bash
gcc c/main.c c/infer.c c/matrix.c c/bmp.c c/preprocess.c -o main.exe -lm
```

### 4. 运行

**批量测试（统计前1000张准确率）：**
```bash
./main.exe
```

**单张 BMP 图片识别：**
```bash
./main.exe test_8.bmp
```

> BMP 图片要求：28×28 像素、24位、无压缩。白底黑字或黑底白字均可（程序自动反色）。

## 准确率

| 模式 | 准确率 |
|---|---|
| Python 训练（测试集10000张） | 97.48% |
| C 推理（测试集前1000张） | 97.50% |
| C 推理 + 图像预处理（用户手绘图片） | ~96% |

## 技术亮点

1. **纯 C 推理**：矩阵运算、激活函数、Softmax 全部手写，零依赖
2. **权重直接编译进程序**：`weights.h` 以 const float 数组形式存储，无需运行时加载
3. **数据增强训练**：随机平移/旋转/缩放，解决全连接网络对位置敏感的问题
4. **自动图像预处理**：裁剪空白→等比缩放→居中，用户随手画的数字也能识别
5. **流式处理**：批量测试时每次只读一张图，内存仅需 784 个 float

## STM32 移植说明

移植时只需以下三个文件（纯运算，无文件 I/O）：
- `c/matrix.c` + `c/matrix.h`
- `c/infer.c` + `c/infer.h`
- `c/weights.h`

`bmp.c`（BMP解析）和 `main.c`（文件读取）是 PC 端验证用，单片机上替换为摄像头/OLED/触摸屏采集像素即可。

模型参数量 101770 个 float，约 400KB Flash，适合 Cortex-M4/M7 级别的单片机。

## License

MIT
