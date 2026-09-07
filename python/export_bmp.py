# -*- coding: utf-8 -*-
"""从MNIST测试集导出第N张图片为28x28 24位BMP，供C推理验证用
用法：py python/export_bmp.py [图片序号] [输出路径]
默认导出第0张（真实标签7）到 test_7.bmp
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "MNIST", "raw")

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "test_%d.bmp" % idx)

# 读取图像
with open(os.path.join(RAW, "t10k-images-idx3-ubyte"), "rb") as f:
    data = f.read()
magic, num, rows, cols = struct.unpack(">IIII", data[:16])
img_start = 16 + idx * rows * cols
img = data[img_start:img_start + rows * cols]

# 读取标签
with open(os.path.join(RAW, "t10k-labels-idx1-ubyte"), "rb") as f:
    ldata = f.read()
label = ldata[8 + idx]

# 构造24位BMP
width, height = 28, 28
row_size = (width * 3 + 3) & ~3   # 每行字节数向上取整到4倍数，28*3=84
pixel_size = row_size * height
file_size = 14 + 40 + pixel_size

bmp = b"BM"
bmp += struct.pack("<I", file_size)
bmp += struct.pack("<HH", 0, 0)
bmp += struct.pack("<I", 54)                       # 像素数据偏移
bmp += struct.pack("<I", 40)                       # 信息头大小
bmp += struct.pack("<ii", width, height)           # 宽高，height正数=从下到上
bmp += struct.pack("<HH", 1, 24)                   # 位面数，位深
bmp += struct.pack("<I", 0)                         # 无压缩
bmp += struct.pack("<I", pixel_size)
bmp += struct.pack("<ii", 2835, 2835)              # 分辨率
bmp += struct.pack("<II", 0, 0)

# 像素数据：BMP从下到上、从左到右、BGR顺序
for y in range(height - 1, -1, -1):
    row = bytearray(row_size)
    for x in range(width):
        gray = img[y * 28 + x]
        row[x * 3]     = gray   # B
        row[x * 3 + 1] = gray   # G
        row[x * 3 + 2] = gray   # R
    bmp += bytes(row)

with open(out_path, "wb") as f:
    f.write(bmp)

print("导出完成:", out_path)
print("测试集第%d张，真实标签: %d" % (idx, label))
