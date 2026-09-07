#include "infer.h"
#include "matrix.h"
#include "weights.h"   /* 训练导出的权重：W1[784*128], b1[128], W2[128*10], b2[10] */

/*
 * 完整前向推理：
 *   input(784) --矩阵乘+偏置--> h1(128) --ReLU--> h1(128)
 *        --矩阵乘+偏置--> out(10) --Softmax--> 概率(10) --Argmax--> 数字
 */
int mnist_infer(const float *input_img)
{
    float h1[128];   /* 隐藏层输出（栈上分配，仅128个float，极小） */
    float out[10];   /* 输出层10个概率 */

    /* ---- 第一层全连接：input * W1 + b1 ---- */
    /* W1已在导出时转置为[784,128]行优先，与matrix_mul_bias要求一致 */
    matrix_mul_bias(input_img, W1, b1, h1, 784, 128);
    /* ReLU激活：负数置0 */
    relu(h1, 128);

    /* ---- 第二层全连接：h1 * W2 + b2 ---- */
    matrix_mul_bias(h1, W2, b2, out, 128, 10);
    /* Softmax：转成0~1概率分布（和为1） */
    softmax(out, 10);

    /* ---- 取最大概率的下标，就是识别的数字 ---- */
    return argmax(out, 10);
}
