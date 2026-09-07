#include "bmp.h"
#include <stdio.h>
#include <stdlib.h>

/* BMP文件头（14字节），按1字节对齐避免编译器填充 */
#pragma pack(push, 1)
typedef struct {
    unsigned char  type[2];      /* "BM" */
    unsigned int   size;         /* 文件总大小 */
    unsigned short reserved1;
    unsigned short reserved2;
    unsigned int   offset;       /* 像素数据起始偏移 */
} BMPFileHeader;

/* BMP信息头（40字节，BITMAPINFOHEADER） */
typedef struct {
    unsigned int   header_size;  /* 40 */
    int            width;
    int            height;       /* 正数=从下到上存储，负数=从上到下 */
    unsigned short planes;       /* 1 */
    unsigned short bit_count;    /* 24 */
    unsigned int   compression;  /* 0=BI_RGB无压缩 */
    unsigned int   image_size;
    int            x_ppm;
    int            y_ppm;
    unsigned int   colors_used;
    unsigned int   colors_important;
} BMPInfoHeader;
#pragma pack(pop)

int bmp_load_to_mnist(const char *filepath, float output[784])
{
    FILE *f = fopen(filepath, "rb");
    if (!f) return -1;

    BMPFileHeader fh;
    BMPInfoHeader  ih;
    if (fread(&fh, sizeof(fh), 1, f) != 1) { fclose(f); return -1; }
    if (fread(&ih, sizeof(ih), 1, f) != 1) { fclose(f); return -1; }

    /* 校验：必须是BM、28x28、24位、无压缩 */
    if (fh.type[0] != 'B' || fh.type[1] != 'M') { fclose(f); return -1; }
    if (ih.width != 28 || abs(ih.height) != 28)  { fclose(f); return -1; }
    if (ih.bit_count != 24)                         { fclose(f); return -1; }
    if (ih.compression != 0)                        { fclose(f); return -1; }

    /* 定位到像素数据 */
    fseek(f, fh.offset, SEEK_SET);

    /* 24位BMP每行字节数必须向上取整到4的倍数；28*3=84已是4倍数 */
    int row_size = (ih.width * 3 + 3) & ~3;
    unsigned char *row_buf = (unsigned char *)malloc(row_size);
    if (!row_buf) { fclose(f); return -1; }

    unsigned char gray[784];      /* 临时存灰度像素 */
    int height = abs(ih.height);
    int bottom_up = (ih.height > 0);  /* BMP默认从下到上存储 */

    for (int y = 0; y < height; y++) {
        fread(row_buf, 1, row_size, f);
        for (int x = 0; x < ih.width; x++) {
            /* BMP像素顺序是 BGR */
            unsigned char b = row_buf[x * 3];
            unsigned char g = row_buf[x * 3 + 1];
            unsigned char r = row_buf[x * 3 + 2];
            /* 灰度化（人眼亮度公式） */
            unsigned char gray_val = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
            /* BMP从下到上存储，翻转为从上到下（与MNIST一致） */
            int dst_y = bottom_up ? (height - 1 - y) : y;
            gray[dst_y * 28 + x] = gray_val;
        }
    }
    free(row_buf);
    fclose(f);

    /* 自动反色：MNIST是黑底白字(背景0)。
       如果图片平均亮度>127，说明是白底黑字，反色变成黑底白字 */
    int sum = 0;
    for (int i = 0; i < 784; i++) sum += gray[i];
    if (sum / 784 > 127) {
        for (int i = 0; i < 784; i++) gray[i] = 255 - gray[i];
    }

    /* 归一化到 0.0~1.0（必须和训练时一致） */
    for (int i = 0; i < 784; i++) output[i] = gray[i] / 255.0f;

    return 0;
}
