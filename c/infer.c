#include "infer.h"
#include "matrix.h"
#include "weights.h"   /* 训练导出的权重：W1[784*128], b1[128], W2[128*10], b2[10] */

/*
 * 完整前向推理，输出10个类别的概率：
 *   input(784) --矩阵乘+偏置--> h1(128) --ReLU--> h1(128)
 *        --矩阵乘+偏置--> out(10) --Softmax--> 概率(10)
 */
MNIST_API void mnist_predict_probs(const float *input_img, float probs[10])
{
    float h1[128];   /* 隐藏层输出（栈上分配，仅128个float，极小） */

    /* ---- 第一层全连接：input * W1 + b1 ---- */
    /* W1已在导出时转置为[784,128]行优先，与matrix_mul_bias要求一致 */
    matrix_mul_bias(input_img, W1, b1, h1, 784, 128);
    /* ReLU激活：负数置0 */
    relu(h1, 128);

    /* ---- 第二层全连接：h1 * W2 + b2 ---- */
    matrix_mul_bias(h1, W2, b2, probs, 128, 10);
    /* Softmax：转成0~1概率分布（和为1） */
    softmax(probs, 10);
}

/* 只返回识别数字（取最大概率下标） */
MNIST_API int mnist_infer(const float *input_img)
{
    float probs[10];
    mnist_predict_probs(input_img, probs);
    return argmax(probs, 10);
}
