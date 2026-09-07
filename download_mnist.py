# -*- coding: utf-8 -*-
"""下载并解压 MNIST 数据集（纯标准库，无需 torch）
下载到 ./data/MNIST/raw/ 目录，与 torchvision 的目录结构一致。
"""
import gzip
import os
import shutil
import urllib.request

# torchvision 官方镜像地址（比官网 yann.lecun.com 更快更稳定）
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "MNIST", "raw")
os.makedirs(ROOT, exist_ok=True)

for name in FILES:
    url = BASE_URL + name
    gz_path = os.path.join(ROOT, name)
    # 已下载且非空则跳过
    if os.path.exists(gz_path) and os.path.getsize(gz_path) > 0:
        print(f"[跳过] {name} 已存在")
        continue
    print(f"[下载] {name} ...")
    try:
        urllib.request.urlretrieve(url, gz_path)
    except Exception as e:
        print(f"[失败] {name} 下载出错: {e}")
        if os.path.exists(gz_path):
            os.remove(gz_path)
        raise
    # 解压 gz，得到 idx3-ubyte / idx1-ubyte 原始文件
    out_path = gz_path[:-3]
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"[完成] {name} -> {os.path.basename(out_path)}")

print("\n全部完成，data/MNIST/raw/ 下文件列表：")
for f in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, f)
    print(f"  {f:35s} {os.path.getsize(p):>12,} bytes")
