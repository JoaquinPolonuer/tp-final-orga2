#include "wave_simulation_common.h"
#include <assert.h>

extern void fft_1d_asm(Complex *x, int n, int inverse);

static void fft2d_asm(Complex *data, int rows, int cols, int inverse)
{

    Complex *temp = (Complex *)malloc(cols * sizeof(Complex));

    for (int i = 0; i < rows; i++)
    {
        memcpy(temp, &data[i * cols], cols * sizeof(Complex));
        fft_1d_asm(temp, cols, inverse);
        memcpy(&data[i * cols], temp, cols * sizeof(Complex));
    }

    temp = (Complex *)realloc(temp, rows * sizeof(Complex));
    for (int j = 0; j < cols; j++)
    {
        for (int i = 0; i < rows; i++)
        {
            temp[i] = data[i * cols + j];
        }
        fft_1d_asm(temp, rows, inverse);
        for (int i = 0; i < rows; i++)
        {
            data[i * cols + j] = temp[i];
        }
    }

    free(temp);
}

static void __attribute__((constructor)) init_fft_backend(void) {
    fft2d = fft2d_asm;
}

PyMODINIT_FUNC PyInit_c_backend_asm(void)
{
    return create_python_module("c_backend_asm", "C backend optimized with FFT in Assembler");
}