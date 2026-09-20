# MNIST 手写数字识别 —— 纯 C 实现推理

Python 训练 + 纯 C 部署的手写数字识别项目。神经网络推理完全用 C 语言手写，不依赖任何深度学习框架，可编译为命令行程序、动态库（DLL）或直接移植到嵌入式平台（如 STM32）。

附带一个 Python(tkinter) 手写桌面应用：鼠标写数字，后台通过 ctypes 实时调用 C 推理引擎。

## 项目结构

```
MNIST.c/
├── c/
│   ├── matrix.h/.c      # 底层运算：矩阵乘+偏置、ReLU、Softmax、Argmax
│   ├── infer.h/.c       # 推理接口：前向传播、输出数字/各类别概率
│   ├── preprocess.h/.c  # 图像预处理：裁剪+缩放+居中（对齐MNIST格式）
│   ├── bmp.h/.c         # BMP图片解析（24位，28x28）
│   ├── mnist_api.h      # 动态库(DLL)导出宏
│   ├── main.c            # 命令行主程序：批量测试 / 单张BMP识别
│   └── weights.h         # 训练导出的权重（101770个float参数）
├── app/
│   └── gui.py            # tkinter手写桌面应用（ctypes调用mnist.dll）
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

命令行版本（批量测试 / BMP识别）：
```bash
gcc c/main.c c/infer.c c/matrix.c c/bmp.c c/preprocess.c -o main.exe -lm
```

桌面应用需要的动态库（DLL）：
```bash
gcc -shared -O2 -o mnist.dll c/infer.c c/matrix.c c/preprocess.c -lm
```

### 4. 命令行运行

**批量测试（统计前1000张准确率）：**
```bash
./main.exe
```

**单张 BMP 图片识别：**
```bash
./main.exe test_8.bmp
```

> BMP 图片要求：28×28 像素、24位、无压缩。白底黑字或黑底白字均可（程序自动反色）。

### 5. 手写桌面应用（GUI）

先按第3步编译出 `mnist.dll`，然后运行：
```bash
py app/gui.py
```

在黑色画布上用鼠标写数字，松开鼠标或停顿即实时识别，右侧显示预测结果和 0~9 各类别概率。仅依赖 Python 标准库 tkinter 与 numpy，无需额外安装。

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
6. **推理引擎可复用**：同一套 C 代码既能编译成命令行 exe，也能编译成 DLL 供 Python GUI 通过 ctypes 实时调用，是典型的"C 核心 + 上层界面"嵌入式架构

## STM32 移植说明

移植时只需以下文件（纯运算，无文件 I/O）：
- `c/matrix.c` + `c/matrix.h`（矩阵与激活函数）
- `c/infer.c` + `c/infer.h`（前向推理）
- `c/preprocess.c` + `c/preprocess.h`（图像标准化，摄像头采集的非标准图像建议保留）
- `c/weights.h`（权重）
- `c/mnist_api.h`（DLL导出宏，单片机下 `MNIST_API` 自动为空，可直接忽略）

`bmp.c`（BMP解析）和 `main.c`（文件读取）是 PC 端验证用，单片机上替换为摄像头/OLED/触摸屏采集像素即可。Python GUI 也是同一套 C 引擎编译成 DLL 后调用，与单片机移植共用核心代码。

模型参数量 101770 个 float，约 400KB Flash，适合 Cortex-M4/M7 级别的单片机。

## License

MIT
