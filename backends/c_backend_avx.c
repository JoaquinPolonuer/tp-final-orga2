#include "wave_simulation_common.h"
#include <assert.h>
#include <immintrin.h>

static Complex complex_sub(Complex a, Complex b)
{
    Complex result = {a.real - b.real, a.imag - b.imag};
    return result;
}

// SIMD complex operations
// Data layout: [real1, imag1, real2, imag2] in __m256d
static inline __m256d complex_add_simd(__m256d z1, __m256d z2)
{
    return _mm256_add_pd(z1, z2);
}

static inline __m256d complex_sub_simd(__m256d z1, __m256d z2)
{
    return _mm256_sub_pd(z1, z2);
}

static inline __m256d complex_mul_simd(__m256d z1, __m256d z2)
{
    // z1 = [a1, b1, a2, b2], z2 = [c1, d1, c2, d2]
    // Result = [(a1*c1-b1*d1), (a1*d1+b1*c1), (a2*c2-b2*d2), (a2*d2+b2*c2)]

    __m256d ac_bd = _mm256_mul_pd(z1, z2);               // [a1*c1, b1*d1, a2*c2, b2*d2]
    __m256d z2_swapped = _mm256_shuffle_pd(z2, z2, 0x5); // [d1, c1, d2, c2]
    __m256d ad_bc = _mm256_mul_pd(z1, z2_swapped);       // [a1*d1, b1*c1, a2*d2, b2*c2]

    __m256d real_parts = _mm256_hsub_pd(ac_bd, ac_bd); // [a1*c1-b1*d1, a1*c1-b1*d1, a2*c2-b2*d2, a2*c2-b2*d2]
    __m256d imag_parts = _mm256_hadd_pd(ad_bc, ad_bc); // [a1*d1+b1*c1, a1*d1+b1*c1, a2*d2+b2*c2, a2*d2+b2*c2]

    return _mm256_unpacklo_pd(real_parts, imag_parts); // [a1*c1-b1*d1, a1*d1+b1*c1, a2*c2-b2*d2, a2*d2+b2*c2]
}

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

static void fft_1d_vectorized(Complex *x, int n, int inverse)
{
    assert(n > 0 && (n & (n - 1)) == 0 && "Input length must be a power of 2");

    bit_reverse(x, n);

    for (int len = 2; len <= n; len <<= 1)
    {
        double angle = 2.0 * M_PI / len * (inverse ? 1 : -1);
        Complex w = {cos(angle), sin(angle)};

        for (int i = 0; i < n; i += len)
        {
            Complex wn = {1.0, 0.0};
            int j;

            // SIMD loop: process 2 complex pairs at once
            for (j = 0; j < (len / 2) - 1; j += 2)
            {
                // Load 2 pairs of complex numbers
                __m256d u = _mm256_loadu_pd((double *)&x[i + j]);           // [u1.real, u1.imag, u2.real, u2.imag]
                __m256d v = _mm256_loadu_pd((double *)&x[i + j + len / 2]); // [v1.real, v1.imag, v2.real, v2.imag]

                // Prepare twiddle factors: wn and wn*w
                Complex wn2 = complex_mul(wn, w);
                __m256d twiddle = _mm256_setr_pd(wn.real, wn.imag, wn2.real, wn2.imag);

                // v = v * twiddle
                v = complex_mul_simd(v, twiddle);

                // Butterfly operations
                __m256d result1 = complex_add_simd(u, v); // u + v
                __m256d result2 = complex_sub_simd(u, v); // u - v

                // Store results
                _mm256_storeu_pd((double *)&x[i + j], result1);
                _mm256_storeu_pd((double *)&x[i + j + len / 2], result2);

                // Update twiddle factor for next iteration
                wn = complex_mul(wn2, w);
            }

            // Handle remaining iterations (scalar)
            for (; j < len / 2; j++)
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

static void fft2d_vectorized(Complex *data, int rows, int cols, int inverse)
{

    Complex *temp = (Complex *)malloc(cols * sizeof(Complex));
    for (int i = 0; i < rows; i++)
    {
        memcpy(temp, &data[i * cols], cols * sizeof(Complex));
        fft_1d_vectorized(temp, cols, inverse);
        memcpy(&data[i * cols], temp, cols * sizeof(Complex));
    }

    temp = (Complex *)realloc(temp, rows * sizeof(Complex));
    for (int j = 0; j < cols; j++)
    {
        for (int i = 0; i < rows; i++)
        {
            temp[i] = data[i * cols + j];
        }
        fft_1d_vectorized(temp, rows, inverse);
        for (int i = 0; i < rows; i++)
        {
            data[i * cols + j] = temp[i];
        }
    }

    free(temp);
}

static void __attribute__((constructor)) init_fft_backend(void) {
    fft2d = fft2d_vectorized;
}

// Method definitions
static PyMethodDef PureCBackendMethods[] = {
    {"create_simulation", c_create_simulation, METH_VARARGS, "Create wave simulation"},
    {"add_wave_source", c_add_wave_source, METH_VARARGS, "Add wave source"},
    {"step_simulation", c_step_simulation, METH_VARARGS, "Step simulation"},
    {"get_intensity", c_get_intensity, METH_VARARGS, "Get wave intensity"},
    {"get_real_part", c_get_real_part, METH_VARARGS, "Get real part of wave"},
    {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef purecbackendmodule = {
    PyModuleDef_HEAD_INIT,
    "c_backend_avx",
    "Pure C backend core functions without NumPy",
    -1,
    PureCBackendMethods};

PyMODINIT_FUNC PyInit_c_backend_avx(void)
{
    return PyModule_Create(&purecbackendmodule);
}