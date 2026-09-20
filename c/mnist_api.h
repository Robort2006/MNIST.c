#ifndef MNIST_API_H
#define MNIST_API_H

/*
 * mnist_api.h —— 动态库(DLL)导出宏
 *
 * 编译DLL时这些函数标记为 dllexport，供Python(ctypes)等外部程序调用；
 * 编译普通exe时该标记无害（不影响功能）。
 * Linux/单片机下 MNIST_API 为空，照常静态编译。
 */
#ifdef _WIN32
    #define MNIST_API __declspec(dllexport)
#else
    #define MNIST_API
#endif

#endif /* MNIST_API_H */
