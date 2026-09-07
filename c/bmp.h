#ifndef BMP_H
#define BMP_H

/*
 * bmp.h —— 24位BMP图片解析（PC端验证专用）
 *
 * 作用：读取一张 28x28 的 24位 BMP 手写数字图片，转成 784 个归一化像素。
 * 自动处理：灰度化、BMP上下翻转、白底黑字自动反色（对齐MNIST的黑底白字）。
 *
 * 【STM32移植说明】此文件不需要移植到单片机。
 * 单片机上像素直接来自摄像头/OLED/触摸屏，不需要解析BMP文件。
 * 移植时只需要 matrix.c + infer.c + weights.h。
 */

/* 从BMP文件读取并解析，输出784个归一化像素(0.0~1.0)
 * filepath: BMP文件路径（必须是28x28 24位无压缩BMP）
 * output:   输出数组，784个float
 * 返回0成功，-1失败
 */
int bmp_load_to_mnist(const char *filepath, float output[784]);

#endif /* BMP_H */
