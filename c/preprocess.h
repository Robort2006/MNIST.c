#ifndef PREPROCESS_H
#define PREPROCESS_H

#include "mnist_api.h"

/*
 * preprocess.h —— 手写数字图像预处理（自动标准化为MNIST风格）
 *
 * 作用：把任意位置、任意大小的手写数字，自动转换成MNIST训练数据的标准格式：
 *       1. 二值化分离数字与背景
 *       2. 找数字边界框，裁剪掉空白
 *       3. 等比缩放到最长边20像素（MNIST标准大小）
 *       4. 居中放置到28x28画布
 *
 * 【为什么需要这个】MNIST训练数据全部经过上述预处理，是居中、标准化大小的数字。
 * 用户随手画的数字可能偏左/偏大/偏小，不预处理直接推理会识别错误。
 *
 * 【STM32移植】此模块可直接移植，纯运算无文件I/O。
 */

/* 输入：784个归一化像素(0.0~1.0)，黑底白字
 * 输出：标准化后的784个归一化像素
 */
MNIST_API void preprocess_normalize(const float input[784], float output[784]);

#endif /* PREPROCESS_H */
