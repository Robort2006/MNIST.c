#include "preprocess.h"
#include <string.h>
//预处理

void preprocess_normalize(const float input[784], float output[784])
{
    /* 1. 输出清零 */
    memset(output, 0, 784 * sizeof(float));

    /* 2. 找最大亮度 */
    float max_val = 0.0f;
    for (int i = 0; i < 784; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    if (max_val < 0.01f) return;   /* 全黑图 */

    /* 3. 行列投影法找边界（比逐像素二值化更鲁棒，能保留抗锯齿边缘）
     *    计算每一行/列的像素和，和超过最大值10%的行/列视为包含数字 */
    float row_sum[28] = {0}, col_sum[28] = {0};
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            float v = input[y * 28 + x];
            row_sum[y] += v;
            col_sum[x] += v;
        }
    }
    float max_row = 0, max_col = 0;
    for (int i = 0; i < 28; i++) {
        if (row_sum[i] > max_row) max_row = row_sum[i];
        if (col_sum[i] > max_col) max_col = col_sum[i];
    }
    float row_thr = max_row * 0.1f;
    float col_thr = max_col * 0.1f;

    int min_y = 28, max_y = -1, min_x = 28, max_x = -1;
    for (int y = 0; y < 28; y++) {
        if (row_sum[y] > row_thr) {
            if (y < min_y) min_y = y;
            if (y > max_y) max_y = y;
        }
    }
    for (int x = 0; x < 28; x++) {
        if (col_sum[x] > col_thr) {
            if (x < min_x) min_x = x;
            if (x > max_x) max_x = x;
        }
    }
    if (max_x < 0 || max_y < 0) return;   /* 没找到数字 */

    int bw = max_x - min_x + 1;   /* 数字实际宽度 */
    int bh = max_y - min_y + 1;   /* 数字实际高度 */

    /* 4. 等比缩放到最长边20像素（MNIST标准） */
    int target_w, target_h;
    if (bw >= bh) {
        target_w = 20;
        target_h = (bh * 20 + bw / 2) / bw;
    } else {
        target_h = 20;
        target_w = (bw * 20 + bh / 2) / bh;
    }
    if (target_w < 1) target_w = 1;
    if (target_h < 1) target_h = 1;

    /* 5. 居中偏移 */
    int offset_x = (28 - target_w) / 2;
    int offset_y = (28 - target_h) / 2;

    /* 6. 最近邻插值缩放 + 居中放置 */
    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            int src_x = min_x + (x * bw + target_w / 2) / target_w;
            int src_y = min_y + (y * bh + target_h / 2) / target_h;
            if (src_x < 0) src_x = 0; if (src_x > 27) src_x = 27;
            if (src_y < 0) src_y = 0; if (src_y > 27) src_y = 27;
            output[(offset_y + y) * 28 + (offset_x + x)] = input[src_y * 28 + src_x];
        }
    }
}
