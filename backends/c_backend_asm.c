#include "wave_simulation_common.h"
#include <assert.h>

// Assembly FFT functions
extern void fft_1d_asm(Complex *x, int n, int inverse);
// extern void test_complex_mul(Complex *a, Complex *b, Complex *result);

// 2D FFT implementation
static void fft2d_asm(Complex *data, int rows, int cols, int inverse)
{

    Complex *temp = (Complex *)malloc(cols * sizeof(Complex));
    // FFT on rows
    for (int i = 0; i < rows; i++)
    {
        memcpy(temp, &data[i * cols], cols * sizeof(Complex));
        fft_1d_asm(temp, cols, inverse);
        memcpy(&data[i * cols], temp, cols * sizeof(Complex));
    }

    // FFT on columns
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

// Initialize the FFT function pointer
static void __attribute__((constructor)) init_fft_backend(void) {
    fft2d = fft2d_asm;
}

// Method definitions
static PyMethodDef PureCBackendMethods[] = {
    {"create_simulation", c_create_simulation, METH_VARARGS, "Create wave simulation"},
    {"add_wave_source", c_add_wave_source, METH_VARARGS, "Add wave source"},
    {"step_simulation", c_step_simulation, METH_VARARGS, "Step simulation"},
    {"get_intensity", c_get_intensity, METH_VARARGS, "Get wave intensity"},
    {"get_real_part", c_get_real_part, METH_VARARGS, "Get real part of wave"},
    // {"test_complex_mul", c_test_complex_mul, METH_VARARGS, "Test complex multiplication macro"},
    {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef purecbackendmodule = {
    PyModuleDef_HEAD_INIT,
    "c_backend_asm",
    "Pure C backend core functions without NumPy",
    -1,
    PureCBackendMethods};

PyMODINIT_FUNC PyInit_c_backend_asm(void)
{
    return PyModule_Create(&purecbackendmodule);
}