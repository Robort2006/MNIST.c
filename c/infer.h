#ifndef INFER_H
#define INFER_H

/*
 * MNIST手写数字推理接口
 * 网络结构：784(输入) -> 128(隐藏,ReLU) -> 10(输出,Softmax)
 *
 * 参数：input_img —— 784个像素，必须已归一化到 0~1（原始像素/255.0）
 * 返回：识别的数字 0~9
 */
int mnist_infer(const float *input_img);

#endif /* INFER_H */
