# -*- coding: utf-8 -*-
"""
gui.py —— MNIST 手写数字识别桌面应用
鼠标在黑色画布上写数字，后台调用纯 C 推理引擎(mnist.dll)实时识别。

技术栈：tkinter(界面) + numpy(画布缩放) + ctypes(调用C编译的DLL)
运行方式（在项目根目录）：
    py app/gui.py
"""
import os
import ctypes
import tkinter as tk
from tkinter import ttk
import numpy as np

# ---------- 路径 ----------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # 项目根目录
DLL_PATH = os.path.join(ROOT, "mnist.dll")

# ---------- 画布参数 ----------
CANVAS_SIZE = 280        # 显示画布 280x280（MNIST 28x28 的 10 倍）
SCALE = CANVAS_SIZE // 28
BRUSH_CORE = 11          # 画笔实心半径（像素）
BRUSH_R = 17             # 画笔软边外半径（抗锯齿）
INK_THRESHOLD = 0.05     # 画布有墨迹的判定阈值


class MNISTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST 手写数字识别 —— Python界面 + 纯C推理")
        self.root.resizable(False, False)

        # 后端画布：280x280 的浮点数组，0=黑底，1=白色墨迹
        self.img = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
        self._after_id = None          # 实时识别的定时器句柄

        self._load_dll()
        self._build_ui()
        self._bind_events()

    # ---------- 加载 C 推理 DLL ----------
    def _load_dll(self):
        self.lib = ctypes.CDLL(DLL_PATH)
        # void preprocess_normalize(const float* in, float* out)
        self.lib.preprocess_normalize.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        self.lib.preprocess_normalize.restype = None
        # void mnist_predict_probs(const float* in, float probs[10])
        self.lib.mnist_predict_probs.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        self.lib.mnist_predict_probs.restype = None

    # ---------- 界面 ----------
    def _build_ui(self):
        bg = "#f5f5f5"
        self.root.configure(bg=bg)

        # 左侧：手写画布
        left = tk.Frame(self.root, bg=bg, padx=14, pady=14)
        left.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(left, text="用鼠标在这里写一个数字（0~9）",
                 bg=bg, font=("Microsoft YaHei", 11)).pack(pady=(0, 8))
        self.canvas = tk.Canvas(left, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", highlightthickness=2,
                                highlightbackground="#888")
        self.canvas.pack()
        tk.Button(left, text="清  除", font=("Microsoft YaHei", 11),
                  width=12, command=self.clear, bg="#d9534f", fg="white",
                  relief=tk.FLAT, pady=6).pack(pady=12)

        # 右侧：结果与概率
        right = tk.Frame(self.root, bg="white", padx=18, pady=14)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(right, text="识别结果", bg="white",
                 font=("Microsoft YaHei", 12), fg="#666").pack(anchor="w")
        self.result_var = tk.StringVar(value="--")
        self.result_label = tk.Label(right, textvariable=self.result_var,
                                     bg="white", fg="#1f6feb",
                                     font=("Arial", 64, "bold"))
        self.result_label.pack(anchor="w", pady=(0, 10))

        tk.Label(right, text="各类别概率", bg="white",
                 font=("Microsoft YaHei", 11), fg="#666").pack(anchor="w")

        # 0~9 每个数字一行：数字标签 + 进度条 + 百分比
        self.bars = []
        self.pct_labels = []
        self.digit_labels = []
        bar_frame = tk.Frame(right, bg="white")
        bar_frame.pack(fill=tk.X, pady=4)
        for d in range(10):
            row = tk.Frame(bar_frame, bg="white")
            row.pack(fill=tk.X, pady=2)
            dl = tk.Label(row, text=str(d), width=2, bg="white",
                          font=("Arial", 11, "bold"), fg="#333")
            dl.pack(side=tk.LEFT)
            bar = ttk.Progressbar(row, length=180, maximum=100)
            bar.pack(side=tk.LEFT, padx=6)
            bar["value"] = 0
            pl = tk.Label(row, text="0.0%", width=7, anchor="e",
                          bg="white", font=("Arial", 9), fg="#555")
            pl.pack(side=tk.LEFT)
            self.digit_labels.append(dl)
            self.bars.append(bar)
            self.pct_labels.append(pl)

        tip = tk.Label(right, text="写完停顿即自动识别，也可松开鼠标识别",
                       bg="white", fg="#999", font=("Microsoft YaHei", 9),
                       wraplength=240, justify=tk.LEFT)
        tip.pack(side=tk.BOTTOM, anchor="w", pady=(12, 0))

    def _bind_events(self):
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.predict())

    # ---------- 画笔 ----------
    def _stamp(self, cx, cy):
        """在(cx,cy)处盖一个软边圆点：同步更新后端数组和显示画布"""
        x0, x1 = max(0, cx - BRUSH_R), min(CANVAS_SIZE - 1, cx + BRUSH_R)
        y0, y1 = max(0, cy - BRUSH_R), min(CANVAS_SIZE - 1, cy + BRUSH_R)
        if x1 < x0 or y1 < y0:
            return
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        # 核心全白，外围线性衰减，形成抗锯齿软边
        val = np.clip(1.0 - (dist - BRUSH_CORE) / (BRUSH_R - BRUSH_CORE),
                      0.0, 1.0)
        region = self.img[y0:y1 + 1, x0:x1 + 1]
        np.maximum(region, val, out=region)
        # 显示层画实心白圆
        r = (BRUSH_CORE + BRUSH_R) // 2
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill="white", outline="white")

    def _on_press(self, event):
        self._stamp(event.x, event.y)

    def _on_motion(self, event):
        self._stamp(event.x, event.y)
        # 防抖：连续书写时每隔一段时间识别一次，实现“实时”
        if self._after_id is None:
            self._after_id = self.root.after(150, self._timed_predict)

    def _timed_predict(self):
        self._after_id = None
        self.predict()

    def clear(self):
        self.img.fill(0.0)
        self.canvas.delete("all")
        self.result_var.set("--")
        for d in range(10):
            self.bars[d]["value"] = 0
            self.pct_labels[d].config(text="0.0%", fg="#555")
            self.digit_labels[d].config(fg="#333")

    # ---------- 调用 C 引擎识别 ----------
    def predict(self):
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

        # 空白画布不识别
        if self.img.max() < INK_THRESHOLD:
            return

        # 280x280 -> 28x28：每 10x10 小块取平均（区域平均，自带抗锯齿）
        small = self.img.reshape(28, SCALE, 28, SCALE).mean(axis=(1, 3))
        flat = small.reshape(-1).astype(np.float32)

        inp = (ctypes.c_float * 784)(*flat)
        norm = (ctypes.c_float * 784)()
        probs = (ctypes.c_float * 10)()

        # 先预处理（裁剪+缩放+居中），再前向推理 —— 与 main.exe 完全一致
        self.lib.preprocess_normalize(inp, norm)
        self.lib.mnist_predict_probs(norm, probs)
        p = np.array(list(probs), dtype=np.float32)
        pred = int(p.argmax())

        # 更新界面
        self.result_var.set(str(pred))
        for d in range(10):
            self.bars[d]["value"] = float(p[d]) * 100.0
            self.pct_labels[d].config(text=f"{p[d] * 100:.1f}%")
            if d == pred:
                self.digit_labels[d].config(fg="#1f6feb")
                self.pct_labels[d].config(fg="#1f6feb")
            else:
                self.digit_labels[d].config(fg="#333")
                self.pct_labels[d].config(fg="#999")


def main():
    if not os.path.exists(DLL_PATH):
        raise FileNotFoundError(
            f"找不到 {DLL_PATH}\n请先在项目根目录编译DLL：\n"
            r"gcc -shared -O2 -o mnist.dll c\infer.c c\matrix.c c\preprocess.c -lm")
    root = tk.Tk()
    MNISTApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
