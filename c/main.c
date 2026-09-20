#include <stdio.h>
#include <stdlib.h>
#include "infer.h"
#include "bmp.h"
#include "preprocess.h"

/*
 * main.c —— MNIST纯C推理主程序
 *
 * 两种运行模式：
 *   1. 单张BMP识别： main.exe <图片.bmp>   （必须是28x28 24位BMP）
 *   2. 批量测试：     main.exe               （读取测试集，统计前1000张准确率）
 *
 * 编译（在项目根目录）：
 *   gcc c\main.c c\infer.c c\matrix.c c\bmp.c -o main.exe -lm
 */

#define TEST_NUM 1000   /* 批量测试时测前1000张 */

/* ---- 单张BMP识别模式 ---- */
static int run_bmp_mode(const char *filepath)
{
    float input[784];
    if (bmp_load_to_mnist(filepath, input) != 0) {
        printf("Error: cannot load BMP file: %s\n", filepath);
        printf("       Make sure it is a 28x28 24-bit uncompressed BMP.\n");
        return 1;
    }
    /* 自动预处理：裁剪+缩放+居中，对齐MNIST训练数据格式 */
    float normalized[784];
    preprocess_normalize(input, normalized);
    int pred = mnist_infer(normalized);
    printf("=== MNIST Pure C Inference ===\n");
    printf("Image:  %s\n", filepath);
    printf("Result: %d\n", pred);
    return 0;
}

/* ---- 批量测试模式（原有功能） ---- */
static int run_batch_mode(void)
{
    FILE *fimg   = fopen("data/MNIST/raw/t10k-images-idx3-ubyte", "rb");
    FILE *flabel = fopen("data/MNIST/raw/t10k-labels-idx1-ubyte", "rb");
    if (!fimg || !flabel) {
        printf("Error: cannot open test data files.\n");
        printf("       Run main.exe from the project root directory.\n");
        return 1;
    }

    fseek(fimg, 16, SEEK_SET);
    fseek(flabel, 8, SEEK_SET);

    unsigned char img_buf[784];
    float input[784];
    int correct = 0;

    printf("=== MNIST Pure C Inference ===\n");
    printf("Testing %d images...\n\n", TEST_NUM);

    for (int i = 0; i < TEST_NUM; i++) {
        fread(img_buf, 1, 784, fimg);
        unsigned char label = (unsigned char)fgetc(flabel);
        for (int j = 0; j < 784; j++) input[j] = img_buf[j] / 255.0f;
        int pred = mnist_infer(input);
        if (pred == label) correct++;
        if ((i + 1) % 100 == 0) {
            printf("  [%4d/%d]  current acc: %.2f%%\n",
                   i + 1, TEST_NUM, correct * 100.0f / (i + 1));
        }
    }
    fclose(fimg);
    fclose(flabel);

    float acc = correct * 100.0f / TEST_NUM;
    printf("\n=== Result ===\n");
    printf("Total:    %d\n", TEST_NUM);
    printf("Correct:  %d\n", correct);
    printf("Wrong:    %d\n", TEST_NUM - correct);
    printf("Accuracy: %.2f%%\n", acc);
    printf("(Python training accuracy was 97.33%%)\n");
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc >= 2) {
        return run_bmp_mode(argv[1]);   /* 单张BMP识别 */
    }
    return run_batch_mode();             /* 批量测试 */
}
