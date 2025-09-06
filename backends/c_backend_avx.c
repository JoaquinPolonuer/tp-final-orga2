#include <Python.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

typedef struct
{
    double real;
    double imag;
} Complex;

typedef struct
{
    Complex *wave;
    Complex *wave_k;
    double *grid_coords;   // Flattened array: [x0,y0,x1,y1,...] for memory efficiency
    double *k_grid_coords; // Flattened array: [kx0,ky0,kx1,ky1,...]
    double *K;             // K magnitude array
    int size;
    double domain_size;
    double wave_speed;
    double dt;
    double dx;
} WaveSimulation;

// Helper functions
static Complex complex_add(Complex a, Complex b)
{
    Complex result = {a.real + b.real, a.imag + b.imag};
    return result;
}

static Complex complex_sub(Complex a, Complex b)
{
    Complex result = {a.real - b.real, a.imag - b.imag};
    return result;
}

static Complex complex_mul(Complex a, Complex b)
{
    Complex result = {
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real};
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

// 1D FFT implementation (Cooley-Tukey). Requires n to be a power of 2.
static void fft_1d_vectorized(Complex *x, int n, int inverse)
{
    // Precondition: n must be a power of 2
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

// 2D FFT implementation
static void fft2d(Complex *data, int rows, int cols, int inverse)
{

    Complex *temp = (Complex *)malloc(cols * sizeof(Complex));
    // FFT on rows
    for (int i = 0; i < rows; i++)
    {
        memcpy(temp, &data[i * cols], cols * sizeof(Complex));
        fft_1d_vectorized(temp, cols, inverse);
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
        fft_1d_vectorized(temp, rows, inverse);
        for (int i = 0; i < rows; i++)
        {
            data[i * cols + j] = temp[i];
        }
    }

    free(temp);
}

// Wave Simulation Functions
static WaveSimulation *create_wave_simulation(int size, double domain_size, double wave_speed, double dt)
{
    WaveSimulation *sim = (WaveSimulation *)malloc(sizeof(WaveSimulation));
    if (!sim)
        return NULL;

    sim->size = size;
    sim->domain_size = domain_size;
    sim->wave_speed = wave_speed;
    sim->dt = dt;
    sim->dx = domain_size / size;

    // Allocate arrays
    sim->wave = (Complex *)calloc(size * size, sizeof(Complex));
    sim->wave_k = (Complex *)calloc(size * size, sizeof(Complex));
    sim->grid_coords = (double *)malloc(size * size * 2 * sizeof(double));
    sim->k_grid_coords = (double *)malloc(size * size * 2 * sizeof(double));
    sim->K = (double *)malloc(size * size * sizeof(double));

    if (!sim->wave || !sim->wave_k || !sim->grid_coords || !sim->k_grid_coords || !sim->K)
    {
        free(sim->wave);
        free(sim->wave_k);
        free(sim->grid_coords);
        free(sim->k_grid_coords);
        free(sim->K);
        free(sim);
        return NULL;
    }

    // Initialize grid coordinates
    double start = -domain_size / 2.0;
    double step = domain_size / size;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            int idx = (i * size + j) * 2;
            sim->grid_coords[idx] = start + j * step;     // x
            sim->grid_coords[idx + 1] = start + i * step; // y
        }
    }

    // Initialize k-grid coordinates
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            int idx = (i * size + j) * 2;
            double kx_freq, ky_freq;

            if (j <= size / 2)
            {
                kx_freq = (double)j / (size * sim->dx);
            }
            else
            {
                kx_freq = (double)(j - size) / (size * sim->dx);
            }

            if (i <= size / 2)
            {
                ky_freq = (double)i / (size * sim->dx);
            }
            else
            {
                ky_freq = (double)(i - size) / (size * sim->dx);
            }

            sim->k_grid_coords[idx] = kx_freq;
            sim->k_grid_coords[idx + 1] = ky_freq;
        }
    }

    // Initialize K magnitude array
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            int idx = i * size + j;
            int coord_idx = idx * 2;
            double kx = sim->k_grid_coords[coord_idx];
            double ky = sim->k_grid_coords[coord_idx + 1];

            double k_mag = sqrt(kx * kx + ky * ky) * 2.0 * M_PI;
            if (i == 0 && j == 0)
            {
                k_mag = 1e-10;
            }

            sim->K[idx] = k_mag;
        }
    }

    return sim;
}

static void wave_sim_add_source(WaveSimulation *sim, double x_pos, double y_pos,
                                double amplitude, double frequency, double width)
{
    for (int i = 0; i < sim->size; i++)
    {
        for (int j = 0; j < sim->size; j++)
        {
            int coord_idx = (i * sim->size + j) * 2;
            double x_val = sim->grid_coords[coord_idx];
            double y_val = sim->grid_coords[coord_idx + 1];

            double r_sq = (x_val - x_pos) * (x_val - x_pos) + (y_val - y_pos) * (y_val - y_pos);
            double envelope = amplitude * exp(-r_sq / (width * width));
            double r = sqrt(r_sq);
            double phase = frequency * r;

            int wave_idx = i * sim->size + j;
            Complex new_val = {envelope * cos(phase), envelope * sin(phase)};
            sim->wave[wave_idx] = complex_add(sim->wave[wave_idx], new_val);
        }
    }

    // Update wave_k using FFT
    memcpy(sim->wave_k, sim->wave, sim->size * sim->size * sizeof(Complex));
    fft2d(sim->wave_k, sim->size, sim->size, 0);
}

static void wave_sim_step(WaveSimulation *sim)
{
    // Apply phase evolution in k-space

    // Time phase evolution
    for (int i = 0; i < sim->size; i++)
    {
        for (int j = 0; j < sim->size; j++)
        {
            int idx = i * sim->size + j;
            double omega = sim->wave_speed * sim->K[idx];
            double phase = -omega * sim->dt;

            Complex phase_factor = {cos(phase), sin(phase)};
            sim->wave_k[idx] = complex_mul(sim->wave_k[idx], phase_factor);
        }
    }

    // Transform back to real space
    memcpy(sim->wave, sim->wave_k, sim->size * sim->size * sizeof(Complex));

    fft2d(sim->wave, sim->size, sim->size, 1);
}

static PyObject *wave_sim_get_intensity(WaveSimulation *sim)
{
    PyObject *result = PyList_New(sim->size);
    for (int i = 0; i < sim->size; i++)
    {
        PyObject *row = PyList_New(sim->size);
        for (int j = 0; j < sim->size; j++)
        {
            int idx = i * sim->size + j;
            Complex val = sim->wave[idx];
            double intensity = val.real * val.real + val.imag * val.imag;
            PyList_SetItem(row, j, PyFloat_FromDouble(intensity));
        }
        PyList_SetItem(result, i, row);
    }
    return result;
}

static PyObject *wave_sim_get_real_part(WaveSimulation *sim)
{
    PyObject *result = PyList_New(sim->size);
    for (int i = 0; i < sim->size; i++)
    {
        PyObject *row = PyList_New(sim->size);
        for (int j = 0; j < sim->size; j++)
        {
            int idx = i * sim->size + j;
            PyList_SetItem(row, j, PyFloat_FromDouble(sim->wave[idx].real));
        }
        PyList_SetItem(result, i, row);
    }
    return result;
}

// Python interface functions
static PyObject *c_create_simulation(PyObject *self, PyObject *args)
{
    int size;
    double domain_size, wave_speed, dt;
    if (!PyArg_ParseTuple(args, "iddd", &size, &domain_size, &wave_speed, &dt))
    {
        return NULL;
    }

    WaveSimulation *sim = create_wave_simulation(size, domain_size, wave_speed, dt);
    if (!sim)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to create simulation");
        return NULL;
    }

    return PyLong_FromVoidPtr(sim);
}

static PyObject *c_add_wave_source(PyObject *self, PyObject *args)
{
    PyObject *ptr_obj;
    double x_pos, y_pos, amplitude, frequency, width;
    if (!PyArg_ParseTuple(args, "Oddddd", &ptr_obj, &x_pos, &y_pos, &amplitude, &frequency, &width))
    {
        return NULL;
    }

    WaveSimulation *sim = (WaveSimulation *)PyLong_AsVoidPtr(ptr_obj);
    wave_sim_add_source(sim, x_pos, y_pos, amplitude, frequency, width);

    Py_RETURN_NONE;
}

static PyObject *c_step_simulation(PyObject *self, PyObject *args)
{
    PyObject *ptr_obj;
    if (!PyArg_ParseTuple(args, "O", &ptr_obj))
    {
        return NULL;
    }

    WaveSimulation *sim = (WaveSimulation *)PyLong_AsVoidPtr(ptr_obj);
    wave_sim_step(sim);

    Py_RETURN_NONE;
}

static PyObject *c_get_intensity(PyObject *self, PyObject *args)
{
    PyObject *ptr_obj;
    if (!PyArg_ParseTuple(args, "O", &ptr_obj))
    {
        return NULL;
    }

    WaveSimulation *sim = (WaveSimulation *)PyLong_AsVoidPtr(ptr_obj);
    return wave_sim_get_intensity(sim);
}

static PyObject *c_get_real_part(PyObject *self, PyObject *args)
{
    PyObject *ptr_obj;
    if (!PyArg_ParseTuple(args, "O", &ptr_obj))
    {
        return NULL;
    }

    WaveSimulation *sim = (WaveSimulation *)PyLong_AsVoidPtr(ptr_obj);
    return wave_sim_get_real_part(sim);
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