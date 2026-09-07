# -*- coding: utf-8 -*-
"""
train.py —— 训练双层全连接网络识别 MNIST，并把权重导出为 C 语言头文件 c/weights.h
网络结构：784(输入) -> 128(隐藏, ReLU) -> 10(输出)
运行方式（在项目根目录 E:\\code\\MNIST.c 下）：
    py python\\train.py
"""
import os
import math
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------- 路径与超参数 ----------
HERE = os.path.dirname(os.path.abspath(__file__))          # train.py 所在的 python 目录
RAW = os.path.join(HERE, "..", "data", "MNIST", "raw")      # 数据集目录
WEIGHTS_H = os.path.join(HERE, "..", "c", "weights.h")      # 导出的 C 头文件

IN_SIZE = 784      # 输入：28*28
HIDDEN = 128       # 隐藏层神经元数
OUT_SIZE = 10      # 输出：数字 0~9
EPOCHS = 10        # 数据增强后需要更多轮收敛
BATCH = 64
LR = 1e-3

# 数据增强幅度
SHIFT_PX = 3       # 随机平移 ±3 像素
ROTATE_DEG = 10    # 随机旋转 ±10 度
SCALE_RATIO = 0.1  # 随机缩放 0.9~1.1 倍


# ---------- 1. 读取 MNIST 原始 idx 文件（不依赖 torchvision） ----------
def load_images(filename):
    """读取图像 idx 文件，返回形状 [N,784]、数值 0~1 的浮点张量"""
    with open(os.path.join(RAW, filename), "rb") as f:
        data = f.read()
    magic, num, rows, cols = struct.unpack(">IIII", data[:16])  # 大端序读文件头
    assert magic == 2051, "图像文件魔数错误"
    x = torch.frombuffer(bytearray(data[16:]), dtype=torch.uint8)
    x = x.reshape(num, rows * cols).float() / 255.0             # 展平 + 归一化到 0~1
    return x


def load_labels(filename):
    """读取标签 idx 文件，返回形状 [N] 的长整型张量"""
    with open(os.path.join(RAW, filename), "rb") as f:
        data = f.read()
    magic, num = struct.unpack(">II", data[:8])
    assert magic == 2049, "标签文件魔数错误"
    y = torch.frombuffer(bytearray(data[8:]), dtype=torch.uint8).long()
    return y


# ---------- 2. 定义双层全连接网络 ----------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_SIZE, HIDDEN)   # 全连接层1：784 -> 128
        self.relu = nn.ReLU()                   # ReLU 激活（负数置0）
        self.fc2 = nn.Linear(HIDDEN, OUT_SIZE)  # 全连接层2：128 -> 10

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)                         # 输出层不接 Softmax，训练时用交叉熵自带
        return x


# ---------- 2.5 数据增强：随机平移+旋转+缩放，让模型对位置/大小/角度鲁棒 ----------
def augment(x):
    """
    x: [N,784] 一个批次的归一化图片
    每张图随机平移、旋转、缩放后返回（形状不变）。
    作用：全连接网络对位置敏感，增强后模型不会因为数字移动几像素就认错。
    """
    n = x.size(0)
    x = x.view(n, 1, 28, 28)

    # 随机平移：像素 -> 归一化坐标（28像素对应2，所以除以14）
    tx = (torch.rand(n) * 2 - 1) * SHIFT_PX / 14.0
    ty = (torch.rand(n) * 2 - 1) * SHIFT_PX / 14.0
    # 随机旋转角度（弧度）
    angle = (torch.rand(n) * 2 - 1) * ROTATE_DEG * math.pi / 180.0
    # 随机缩放系数
    scale = 1.0 + (torch.rand(n) * 2 - 1) * SCALE_RATIO

    cos = torch.cos(angle) / scale
    sin = torch.sin(angle) / scale

    # 组装仿射变换矩阵 [N,2,3]
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, x.size(), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False)
    return x.view(n, 784)


# ---------- 3. 训练 ----------
def train(model, X, y):
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()          # 交叉熵损失（内部已含 Softmax）
    model.train()
    for ep in range(EPOCHS):
        epoch_loss = 0.0
        for xb, yb in loader:                  # 每次取一个小批次
            xb = augment(xb)                   # 数据增强：随机平移/旋转/缩放
            optimizer.zero_grad()              # 清空上一轮梯度
            out = model(xb)                    # 前向
            loss = criterion(out, yb)          # 算损失
            loss.backward()                    # 反向传播
            optimizer.step()                   # 更新权重
            epoch_loss += loss.item() * len(xb)
        print(f"  Epoch {ep+1}/{EPOCHS}  平均损失 {epoch_loss/len(X):.4f}")


# ---------- 4. 测试集评估准确率 ----------
@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    pred = model(X).argmax(dim=1)              # 取10个输出里最大的下标作为预测数字
    acc = (pred == y).float().mean().item()
    return acc


# ---------- 5. 导出权重为 C 头文件（最关键：矩阵转置） ----------
def export_weights(model, path):
    # PyTorch 的 nn.Linear 权重形状是 [out, in]；
    # 而 C 里矩阵乘法 y[j]=Σ x[i]*W[i][j] 需要 [in, out]，所以这里必须 .t() 转置。
    w1 = model.fc1.weight.detach().t().contiguous().reshape(-1)  # [784*128]
    b1 = model.fc1.bias.detach().reshape(-1)                     # [128]
    w2 = model.fc2.weight.detach().t().contiguous().reshape(-1)  # [128*10]
    b2 = model.fc2.bias.detach().reshape(-1)                     # [10]

    def write_array(f, name, tensor):
        f.write(f"const float {name}[{tensor.numel()}] = {{\n")
        vals = tensor.tolist()
        for i, v in enumerate(vals):
            f.write(f"{v:.6f}f,")
            if (i + 1) % 8 == 0:              # 每行8个数，方便阅读
                f.write("\n")
        f.write("\n};\n\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        write_array(f, "W1", w1)
        write_array(f, "b1", b1)
        write_array(f, "W2", w2)
        write_array(f, "b2", b2)
        f.write("#endif // WEIGHTS_H\n")


# ---------- 主流程 ----------
def main():
    torch.manual_seed(0)                       # 固定随机种子，结果可复现
    print("[1/4] 加载 MNIST 数据 ...")
    X_train = load_images("train-images-idx3-ubyte")
    y_train = load_labels("train-labels-idx1-ubyte")
    X_test = load_images("t10k-images-idx3-ubyte")
    y_test = load_labels("t10k-labels-idx1-ubyte")
    print(f"  训练集 {X_train.shape[0]} 张，测试集 {X_test.shape[0]} 张")

    print("[2/4] 开始训练 ...")
    model = MLP()
    train(model, X_train, y_train)

    print("[3/4] 测试集评估 ...")
    acc = evaluate(model, X_test, y_test)
    print(f"  测试集准确率：{acc*100:.2f}%")

    print("[4/4] 导出权重到 c/weights.h ...")
    export_weights(model, WEIGHTS_H)
    size_kb = os.path.getsize(WEIGHTS_H) / 1024
    print(f"  导出完成，文件大小 {size_kb:.0f} KB")
    print("全部结束，接下来可以写 C 推理代码了。")


#如果直接运行这个文件（py train.py），就执行 main()。如果是被别的文件 import，就不执行。
if __name__ == "__main__":
    main()
