#include "wave_simulation_common.h"
#include <assert.h>

// Helper function for complex subtraction (used only in FFT)
static Complex complex_sub(Complex a, Complex b)
{
    Complex result = {a.real - b.real, a.imag - b.imag};
    return result;
}

// Bit reversal for FFT
static void bit_reverse(Complex *x, int n)
{
    int j = 0;
    for (int i = 1; i < n; i++)
    {
        int bit = n >> 1;
        while (j & bit)
        {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j)
        {
            Complex temp = x[i];
            x[i] = x[j];
            x[j] = temp;
        }
    }
}

static void fft_1d(Complex *x, int n, int inverse)
{
    assert(n > 0 && (n & (n - 1)) == 0 && "La longitud debe ser potencia de 2");

    bit_reverse(x, n);

    for (int len = 2; len <= n; len <<= 1)
    {
        double angle = 2.0 * M_PI / len * (inverse ? 1 : -1);
        Complex w = {cos(angle), sin(angle)};

        for (int i = 0; i < n; i += len)
        {
            Complex wn = {1.0, 0.0};
            for (int j = 0; j < len / 2; j++)
            {
                Complex u = x[i + j];
                Complex v = complex_mul(x[i + j + len / 2], wn);
                x[i + j] = complex_add(u, v);
                x[i + j + len / 2] = complex_sub(u, v);
                wn = complex_mul(wn, w);
            }
        }
    }

    if (inverse)
    {
        for (int i = 0; i < n; i++)
        {
            x[i].real /= n;
            x[i].imag /= n;
        }
    }
}

static void fft2d_basic(Complex *data, int rows, int cols, int inverse)
{

    Complex *temp = (Complex *)malloc(cols * sizeof(Complex));

    for (int i = 0; i < rows; i++)
    {
        memcpy(temp, &data[i * cols], cols * sizeof(Complex));
        fft_1d(temp, cols, inverse);
        memcpy(&data[i * cols], temp, cols * sizeof(Complex));
    }

    temp = (Complex *)realloc(temp, rows * sizeof(Complex));
    for (int j = 0; j < cols; j++)
    {
        for (int i = 0; i < rows; i++)
        {
            temp[i] = data[i * cols + j];
        }
        fft_1d(temp, rows, inverse);
        for (int i = 0; i < rows; i++)
        {
            data[i * cols + j] = temp[i];
        }
    }

    free(temp);
}

static void __attribute__((constructor)) init_fft_backend(void)
{
    fft2d = fft2d_basic;
}

static PyMethodDef PureCBackendMethods[] = {
    {"create_simulation", c_create_simulation, METH_VARARGS, "Create wave simulation"},
    {"add_wave_source", c_add_wave_source, METH_VARARGS, "Add wave source"},
    {"step_simulation", c_step_simulation, METH_VARARGS, "Step simulation"},
    {"get_intensity", c_get_intensity, METH_VARARGS, "Get wave intensity"},
    {"get_real_part", c_get_real_part, METH_VARARGS, "Get real part of wave"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef purecbackendmodule = {
    PyModuleDef_HEAD_INIT,
    "c_backend",
    "Pure C backend core functions without NumPy",
    -1,
    PureCBackendMethods};

PyMODINIT_FUNC PyInit_c_backend(void)
{
    return PyModule_Create(&purecbackendmodule);
}