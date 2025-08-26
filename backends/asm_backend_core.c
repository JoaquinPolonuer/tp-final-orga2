#include <Python.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct
{
    double real;
    double imag;
} Complex;

// Inline assembly macros for complex operations (Intel syntax)
static inline void asm_complex_add(Complex *a, Complex *b, Complex *result) {
    __asm__ volatile (
        ".intel_syntax noprefix   \n\t"
        "movupd  xmm0, [%1]       \n\t"  // Load a
        "addpd   xmm0, [%2]       \n\t"  // Add b
        "movupd  [%0], xmm0       \n\t"  // Store result
        ".att_syntax prefix       \n\t"
        :
        : "r" (result), "r" (a), "r" (b)
        : "xmm0", "memory"
    );
}

static inline void asm_complex_sub(Complex *a, Complex *b, Complex *result) {
    __asm__ volatile (
        ".intel_syntax noprefix   \n\t"
        "movupd  xmm0, [%1]       \n\t"  // Load a
        "subpd   xmm0, [%2]       \n\t"  // Subtract b
        "movupd  [%0], xmm0       \n\t"  // Store result
        ".att_syntax prefix       \n\t"
        :
        : "r" (result), "r" (a), "r" (b)
        : "xmm0", "memory"
    );
}

static inline void asm_complex_mul(Complex *a, Complex *b, Complex *result) {
    __asm__ volatile (
        ".intel_syntax noprefix   \n\t"
        "movupd  xmm0, [%1]       \n\t"  // xmm0 = [ar, ai]
        "movupd  xmm1, [%2]       \n\t"  // xmm1 = [br, bi]
        "movapd  xmm2, xmm1       \n\t"  // xmm2 = [br, bi]
        "movapd  xmm3, xmm0       \n\t"  // xmm3 = [ar, ai]
        "unpcklpd xmm1, xmm1      \n\t"  // xmm1 = [br, br]
        "unpckhpd xmm2, xmm2      \n\t"  // xmm2 = [bi, bi]
        "mulpd   xmm0, xmm1       \n\t"  // xmm0 = [ar*br, ai*br]
        "mulpd   xmm3, xmm2       \n\t"  // xmm3 = [ar*bi, ai*bi]
        "shufpd  xmm3, xmm3, 0x1 \n\t"  // xmm3 = [ai*bi, ar*bi]
        "addsubpd xmm0, xmm3      \n\t"  // xmm0 = [ar*br-ai*bi, ai*br+ar*bi]
        "movupd  [%0], xmm0       \n\t"  // Store result
        ".att_syntax prefix       \n\t"
        :
        : "r" (result), "r" (a), "r" (b)
        : "xmm0", "xmm1", "xmm2", "xmm3", "memory"
    );
}

// Convert Python list of lists to C array
static Complex *python_to_c_array(PyObject *py_list, int *rows, int *cols)
{
    if (!PyList_Check(py_list))
    {
        return NULL;
    }

    *rows = (int)PyList_Size(py_list);
    if (*rows == 0)
        return NULL;

    PyObject *first_row = PyList_GetItem(py_list, 0);
    if (!PyList_Check(first_row))
    {
        return NULL;
    }

    *cols = (int)PyList_Size(first_row);
    Complex *array = (Complex *)malloc((*rows) * (*cols) * sizeof(Complex));

    for (int i = 0; i < *rows; i++)
    {
        PyObject *row = PyList_GetItem(py_list, i);
        for (int j = 0; j < *cols; j++)
        {
            PyObject *item = PyList_GetItem(row, j);
            if (PyComplex_Check(item))
            {
                array[i * (*cols) + j].real = PyComplex_RealAsDouble(item);
                array[i * (*cols) + j].imag = PyComplex_ImagAsDouble(item);
            }
            else if (PyFloat_Check(item))
            {
                array[i * (*cols) + j].real = PyFloat_AsDouble(item);
                array[i * (*cols) + j].imag = 0.0;
            }
            else if (PyLong_Check(item))
            {
                array[i * (*cols) + j].real = (double)PyLong_AsLong(item);
                array[i * (*cols) + j].imag = 0.0;
            }
        }
    }

    return array;
}

// Convert C array to Python list of lists
static PyObject *c_to_python_array(Complex *array, int rows, int cols)
{
    PyObject *py_list = PyList_New(rows);

    for (int i = 0; i < rows; i++)
    {
        PyObject *row = PyList_New(cols);
        for (int j = 0; j < cols; j++)
        {
            Complex val = array[i * cols + j];
            PyObject *complex_val = PyComplex_FromDoubles(val.real, val.imag);
            PyList_SetItem(row, j, complex_val);
        }
        PyList_SetItem(py_list, i, row);
    }

    return py_list;
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

// Vectorized FFT butterfly operation using SIMD
static inline void vectorized_butterfly(Complex *x, int i, int j, int len_half, 
                                       double wr, double wi, double wr2, double wi2)
{
    __asm__ volatile (
        // Load twiddle factors into registers
        "movsd   %4, %%xmm4       \n\t"  // xmm4 = [0, wr]
        "movsd   %5, %%xmm5       \n\t"  // xmm5 = [0, wi]  
        "movsd   %6, %%xmm6       \n\t"  // xmm6 = [0, wr2]
        "movsd   %7, %%xmm7       \n\t"  // xmm7 = [0, wi2]
        "unpcklpd %%xmm5, %%xmm4  \n\t"  // xmm4 = [wi, wr]
        "unpcklpd %%xmm7, %%xmm6  \n\t"  // xmm6 = [wi2, wr2]
        
        // Load x[i+j] and x[i+j+1] (2 complex numbers)
        "movupd  (%0), %%xmm0     \n\t"  // xmm0 = [x[i+j].imag, x[i+j].real]
        "movupd  16(%0), %%xmm1   \n\t"  // xmm1 = [x[i+j+1].imag, x[i+j+1].real]
        
        // Load x[i+j+len/2] and x[i+j+len/2+1] 
        "movupd  (%1), %%xmm2     \n\t"  // xmm2 = [x[i+j+len/2].imag, x[i+j+len/2].real]
        "movupd  16(%1), %%xmm3   \n\t"  // xmm3 = [x[i+j+len/2+1].imag, x[i+j+len/2+1].real]
        
        // Save copies for butterfly
        "movapd  %%xmm0, %%xmm8   \n\t"  // Save x[i+j] 
        "movapd  %%xmm1, %%xmm9   \n\t"  // Save x[i+j+1]
        "movapd  %%xmm2, %%xmm10  \n\t"  // Copy for multiplication
        "movapd  %%xmm3, %%xmm11  \n\t"  // Copy for multiplication
        
        // First complex multiplication: xmm2 * xmm4 (w)
        "movapd  %%xmm4, %%xmm12  \n\t"  // Copy w
        "movapd  %%xmm10, %%xmm13 \n\t"  // Copy x[i+j+len/2]
        "unpcklpd %%xmm12, %%xmm12 \n\t"  // xmm12 = [wr, wr]
        "unpckhpd %%xmm4, %%xmm4  \n\t"  // xmm4 = [wi, wi]
        "mulpd   %%xmm12, %%xmm10 \n\t"  // [real*wr, imag*wr]
        "mulpd   %%xmm4, %%xmm13  \n\t"  // [real*wi, imag*wi] 
        "shufpd  $0x1, %%xmm13, %%xmm13 \n\t" // [imag*wi, real*wi]
        "addsubpd %%xmm13, %%xmm10 \n\t"  // [real*wr-imag*wi, imag*wr+real*wi]
        
        // Second complex multiplication: xmm3 * xmm6 (w2)
        "movapd  %%xmm6, %%xmm12  \n\t"  // Copy w2
        "movapd  %%xmm11, %%xmm13 \n\t"  // Copy x[i+j+len/2+1]
        "unpcklpd %%xmm12, %%xmm12 \n\t"  // xmm12 = [wr2, wr2]
        "unpckhpd %%xmm6, %%xmm6  \n\t"  // xmm6 = [wi2, wi2]
        "mulpd   %%xmm12, %%xmm11 \n\t"  // [real*wr2, imag*wr2]
        "mulpd   %%xmm6, %%xmm13  \n\t"  // [real*wi2, imag*wi2]
        "shufpd  $0x1, %%xmm13, %%xmm13 \n\t" // [imag*wi2, real*wi2]
        "addsubpd %%xmm13, %%xmm11 \n\t"  // [real*wr2-imag*wi2, imag*wr2+real*wi2]
        
        // Butterfly operations
        "addpd   %%xmm10, %%xmm0  \n\t"  // x[i+j] = x[i+j] + v1
        "addpd   %%xmm11, %%xmm1  \n\t"  // x[i+j+1] = x[i+j+1] + v2
        "subpd   %%xmm10, %%xmm8  \n\t"  // x[i+j+len/2] = x[i+j] - v1
        "subpd   %%xmm11, %%xmm9  \n\t"  // x[i+j+len/2+1] = x[i+j+1] - v2
        
        // Store results
        "movupd  %%xmm0, (%0)     \n\t"  // Store x[i+j]
        "movupd  %%xmm1, 16(%0)   \n\t"  // Store x[i+j+1] 
        "movupd  %%xmm8, (%1)     \n\t"  // Store x[i+j+len/2]
        "movupd  %%xmm9, 16(%1)   \n\t"  // Store x[i+j+len/2+1]
        
        :
        : "r" (&x[i + j]), "r" (&x[i + j + len_half]), 
          "r" (i), "r" (j),
          "m" (wr), "m" (wi), "m" (wr2), "m" (wi2)
        : "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
          "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "memory"
    );
}

// Vectorized 1D FFT implementation
static void fft_1d(Complex *x, int n, int inverse)
{
    // Ensure n is power of 2
    int power = 1;
    while (power < n)
        power <<= 1;

    if (power != n)
    {
        // Pad with zeros
        for (int i = n; i < power; i++)
        {
            x[i].real = 0.0;
            x[i].imag = 0.0;
        }
        n = power;
    }

    bit_reverse(x, n);

    // Vectorized FFT loops - process 2 butterflies at once
    for (int len = 2; len <= n; len <<= 1)
    {
        double angle = 2.0 * M_PI / len * (inverse ? 1 : -1);
        Complex w = {cos(angle), sin(angle)};
        int len_half = len / 2;

        for (int i = 0; i < n; i += len)
        {
            Complex wn = {1.0, 0.0};
            
            // Process pairs of butterflies with SIMD
            int j;
            for (j = 0; j < len_half - 1; j += 2)
            {
                // Calculate two twiddle factors at once
                double wr1 = wn.real;
                double wi1 = wn.imag;
                
                // wn2 = wn * w
                double wr2 = wr1 * w.real - wi1 * w.imag;
                double wi2 = wr1 * w.imag + wi1 * w.real;
                
                // Vectorized butterfly for j and j+1
                vectorized_butterfly(x, i, j, len_half, wr1, wi1, wr2, wi2);
                
                // Update wn for next iteration: wn *= w^2
                Complex w_squared = {w.real * w.real - w.imag * w.imag,
                                   2.0 * w.real * w.imag};
                Complex temp = {wn.real * w_squared.real - wn.imag * w_squared.imag,
                              wn.real * w_squared.imag + wn.imag * w_squared.real};
                wn = temp;
            }
            
            // Handle remaining butterfly if len_half is odd
            if (j < len_half)
            {
                Complex v;
                asm_complex_mul(&x[i + j + len_half], &wn, &v);
                asm_complex_add(&x[i + j], &v, &x[i + j]);
                asm_complex_sub(&x[i + j], &v, &x[i + j + len_half]);
            }
        }
    }

    if (inverse)
    {
        // Vectorized normalization
        double inv_n = 1.0 / n;
        size_t byte_count = n * 16;
        __asm__ volatile (
            "movsd   %1, %%xmm0       \n\t"
            "unpcklpd %%xmm0, %%xmm0  \n\t"  // xmm0 = [1/n, 1/n]
            "movq    $0, %%rax        \n\t"
            "1:                       \n\t"
            "movupd  (%%rax,%0), %%xmm1 \n\t"
            "mulpd   %%xmm0, %%xmm1   \n\t"
            "movupd  %%xmm1, (%%rax,%0) \n\t"
            "addq    $16, %%rax       \n\t"
            "cmpq    %2, %%rax        \n\t"
            "jl      1b               \n\t"
            :
            : "r" (x), "m" (inv_n), "r" (byte_count)
            : "rax", "xmm0", "xmm1", "memory"
        );
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
        fft_1d(temp, cols, inverse);
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
        fft_1d(temp, rows, inverse);
        for (int i = 0; i < rows; i++)
        {
            data[i * cols + j] = temp[i];
        }
    }

    free(temp);
}

static PyObject *c_fft2(PyObject *self, PyObject *args)
{
    PyObject *input_obj;
    if (!PyArg_ParseTuple(args, "O", &input_obj))
    {
        return NULL;
    }

    int rows, cols;
    Complex *data = python_to_c_array(input_obj, &rows, &cols);
    if (data == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Expected list of lists");
        return NULL;
    }

    fft2d(data, rows, cols, 0);

    PyObject *result = c_to_python_array(data, rows, cols);

    free(data);

    return result;
}

static PyObject *c_ifft2(PyObject *self, PyObject *args)
{
    PyObject *input_obj;
    if (!PyArg_ParseTuple(args, "O", &input_obj))
    {
        return NULL;
    }

    int rows, cols;
    Complex *data = python_to_c_array(input_obj, &rows, &cols);
    if (data == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Expected list of lists");
        return NULL;
    }

    fft2d(data, rows, cols, 1);

    PyObject *result = c_to_python_array(data, rows, cols);

    free(data);

    return result;
}

static PyObject *c_fftfreq(PyObject *self, PyObject *args)
{
    int n;
    double d = 1.0;
    if (!PyArg_ParseTuple(args, "i|d", &n, &d))
    {
        return NULL;
    }

    PyObject *py_list = PyList_New(n);

    for (int i = 0; i < n; i++)
    {
        double freq;
        if (i <= n / 2)
        {
            freq = (double)i / (n * d);
        }
        else
        {
            freq = (double)(i - n) / (n * d);
        }
        PyList_SetItem(py_list, i, PyFloat_FromDouble(freq));
    }

    return py_list;
}

static PyObject *c_abs_array(PyObject *self, PyObject *args)
{
    PyObject *input_obj;
    if (!PyArg_ParseTuple(args, "O", &input_obj))
    {
        return NULL;
    }

    int rows, cols;
    Complex *data = python_to_c_array(input_obj, &rows, &cols);
    if (data == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Expected list of lists");
        return NULL;
    }

    PyObject *result = PyList_New(rows);
    for (int i = 0; i < rows; i++)
    {
        PyObject *row = PyList_New(cols);
        for (int j = 0; j < cols; j++)
        {
            Complex val = data[i * cols + j];
            double abs_val = sqrt(val.real * val.real + val.imag * val.imag);
            PyList_SetItem(row, j, PyFloat_FromDouble(abs_val));
        }
        PyList_SetItem(result, i, row);
    }

    free(data);
    return result;
}

static PyObject *c_real_array(PyObject *self, PyObject *args)
{
    PyObject *input_obj;
    if (!PyArg_ParseTuple(args, "O", &input_obj))
    {
        return NULL;
    }

    int rows, cols;
    Complex *data = python_to_c_array(input_obj, &rows, &cols);
    if (data == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Expected list of lists");
        return NULL;
    }

    PyObject *result = PyList_New(rows);
    for (int i = 0; i < rows; i++)
    {
        PyObject *row = PyList_New(cols);
        for (int j = 0; j < cols; j++)
        {
            Complex val = data[i * cols + j];
            PyList_SetItem(row, j, PyFloat_FromDouble(val.real));
        }
        PyList_SetItem(result, i, row);
    }

    free(data);
    return result;
}

// Method definitions
static PyMethodDef PureCBackendMethods[] = {
    {"fft2", c_fft2, METH_VARARGS, "2D FFT"},
    {"ifft2", c_ifft2, METH_VARARGS, "2D IFFT"},
    {"fftfreq", c_fftfreq, METH_VARARGS, "FFT frequency array"},
    {"abs_array", c_abs_array, METH_VARARGS, "Absolute value of array"},
    {"real_array", c_real_array, METH_VARARGS, "Real part of array"},
    {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef asmbackendmodule = {
    PyModuleDef_HEAD_INIT,
    "asm_backend_core",
    "ASM-enhanced backend core functions",
    -1,
    PureCBackendMethods};

PyMODINIT_FUNC PyInit_asm_backend_core(void)
{
    return PyModule_Create(&asmbackendmodule);
}