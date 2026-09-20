#ifndef INFER_H
#define INFER_H

#include "mnist_api.h"

/*
 * MNIST手写数字推理接口
 * 网络结构：784(输入) -> 128(隐藏,ReLU) -> 10(输出,Softmax)
 *
 * 参数：input_img —— 784个像素，必须已归一化到 0~1（原始像素/255.0）
 * 返回：识别的数字 0~9
 */
MNIST_API int mnist_infer(const float *input_img);

/* 完整推理并输出10个类别的概率（供GUI显示置信度）
 * input_img : 784个归一化像素
 * probs     : 输出数组，长度10，每个元素0~1，和为1
 */
MNIST_API void mnist_predict_probs(const float *input_img, float probs[10]);

#endif /* INFER_H */
