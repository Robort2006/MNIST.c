#include "matrix.h"
#include <math.h>

/* 矩阵乘加偏置：y[j] = b[j] + Σ_i x[i] * W[i*w_cols + j] */
void matrix_mul_bias(const float *x, const float *W, const float *b,
                     float *y, int x_len, int w_cols)
{
    for (int j = 0; j < w_cols; j++) {
        y[j] = b[j];                          /* 先加偏置 */
        for (int i = 0; i < x_len; i++) {
            y[j] += x[i] * W[i * w_cols + j]; /* 累加 x[i]*W[i][j] */
        }
    }
}

/* ReLU：负数置0 */
void relu(float *vec, int len)
{
    for (int i = 0; i < len; i++) {
        if (vec[i] < 0.0f) {
            vec[i] = 0.0f;
        }
    }
}

/* Softmax：vec[i] = exp(vec[i]) / Σ exp(vec[j])
 * 先减最大值，防止指数运算溢出（数值稳定技巧） */
void softmax(float *vec, int len)
{
    /* 找最大值 */
    float maxv = vec[0];
    for (int i = 1; i < len; i++) {
        if (vec[i] > maxv) maxv = vec[i];
    }
    /* 每个元素减最大值后取exp，同时累加求和 */
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        vec[i] = expf(vec[i] - maxv);
        sum += vec[i];
    }
    /* 除以总和，得到概率 */
    for (int i = 0; i < len; i++) {
        vec[i] /= sum;
    }
}

/* Argmax：返回最大值的下标 */
int argmax(const float *vec, int len)
{
    int idx = 0;
    float maxv = vec[0];
    for (int i = 1; i < len; i++) {
        if (vec[i] > maxv) {
            maxv = vec[i];
            idx = i;
        }
    }
    return idx;
}
