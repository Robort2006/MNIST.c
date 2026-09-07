#ifndef MATRIX_H
#define MATRIX_H

/*
 * 底层运算库：神经网络前向传播的4个基本运算
 * 所有矩阵按行优先存储（C语言默认），与Python导出的weights.h一致
 */

/* 矩阵乘加偏置：y = x * W + b
 * x: [1, x_len]   W: [x_len, w_cols]   b: [w_cols]   y: [1, w_cols]
 * W的索引：W[i * w_cols + j]  表示第i行第j列
 */
void matrix_mul_bias(const float *x, const float *W, const float *b,
                     float *y, int x_len, int w_cols);

/* ReLU激活：负数置0，正数不变 */
void relu(float *vec, int len);

/* Softmax：把输出转成概率分布（所有元素和为1）
 * 内部做了减最大值处理，防止exp溢出 */
void softmax(float *vec, int len);

/* Argmax：返回数组中最大值的下标 */
int argmax(const float *vec, int len);

#endif /* MATRIX_H */
