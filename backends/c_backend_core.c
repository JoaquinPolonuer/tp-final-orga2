#include <Python.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

typedef struct {
    double real;
    double imag;
} Complex;

// Helper functions
static Complex complex_add(Complex a, Complex b) {
    Complex result = {a.real + b.real, a.imag + b.imag};
    return result;
}

static Complex complex_sub(Complex a, Complex b) {
    Complex result = {a.real - b.real, a.imag - b.imag};
    return result;
}

static Complex complex_mul(Complex a, Complex b) {
    Complex result = {
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
    return result;
}

static Complex complex_exp(Complex z) {
    double exp_real = exp(z.real);
    Complex result = {
        exp_real * cos(z.imag),
        exp_real * sin(z.imag)
    };
    return result;
}

static double complex_abs(Complex z) {
    return sqrt(z.real * z.real + z.imag * z.imag);
}

// Bit reversal for FFT
static void bit_reverse(Complex *x, int n) {
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            Complex temp = x[i];
            x[i] = x[j];
            x[j] = temp;
        }
    }
}

// 1D FFT implementation (Cooley-Tukey)
static void fft_1d(Complex *x, int n, int inverse) {
    // Ensure n is power of 2
    int power = 1;
    while (power < n) power <<= 1;
    
    if (power != n) {
        // Pad with zeros
        for (int i = n; i < power; i++) {
            x[i].real = 0.0;
            x[i].imag = 0.0;
        }
        n = power;
    }
    
    bit_reverse(x, n);
    
    for (int len = 2; len <= n; len <<= 1) {
        double angle = 2.0 * M_PI / len * (inverse ? 1 : -1);
        Complex w = {cos(angle), sin(angle)};
        
        for (int i = 0; i < n; i += len) {
            Complex wn = {1.0, 0.0};
            for (int j = 0; j < len / 2; j++) {
                Complex u = x[i + j];
                Complex v = complex_mul(x[i + j + len / 2], wn);
                x[i + j] = complex_add(u, v);
                x[i + j + len / 2] = complex_sub(u, v);
                wn = complex_mul(wn, w);
            }
        }
    }
    
    if (inverse) {
        for (int i = 0; i < n; i++) {
            x[i].real /= n;
            x[i].imag /= n;
        }
    }
}

// 2D FFT implementation
static void fft2d(Complex *data, int rows, int cols, int inverse) {
    Complex *temp = (Complex*)malloc(cols * sizeof(Complex));
    
    // FFT on rows
    for (int i = 0; i < rows; i++) {
        memcpy(temp, &data[i * cols], cols * sizeof(Complex));
        fft_1d(temp, cols, inverse);
        memcpy(&data[i * cols], temp, cols * sizeof(Complex));
    }
    
    // FFT on columns
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            temp[i] = data[i * cols + j];
        }
        fft_1d(temp, rows, inverse);
        for (int i = 0; i < rows; i++) {
            data[i * cols + j] = temp[i];
        }
    }
    
    free(temp);
}

// Python wrapper functions
static PyObject* c_zeros(PyObject* self, PyObject* args) {
    int rows, cols;
    if (!PyArg_ParseTuple(args, "ii", &rows, &cols)) {
        return NULL;
    }
    
    npy_intp dims[2] = {rows, cols};
    PyObject* array = PyArray_ZEROS(2, dims, NPY_CDOUBLE, 0);
    return array;
}

static PyObject* c_linspace(PyObject* self, PyObject* args) {
    double start, stop;
    int num;
    if (!PyArg_ParseTuple(args, "ddi", &start, &stop, &num)) {
        return NULL;
    }
    
    npy_intp dims[1] = {num};
    PyObject* array = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    double* data = (double*)PyArray_DATA((PyArrayObject*)array);
    
    if (num == 1) {
        data[0] = start;
    } else {
        double step = (stop - start) / (num - 1);
        for (int i = 0; i < num; i++) {
            data[i] = start + i * step;
        }
    }
    
    return array;
}

static PyObject* c_meshgrid(PyObject* self, PyObject* args) {
    PyObject* x_obj, *y_obj;
    if (!PyArg_ParseTuple(args, "OO", &x_obj, &y_obj)) {
        return NULL;
    }
    
    PyArrayObject* x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if (x_array == NULL || y_array == NULL) {
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        return NULL;
    }
    
    int x_len = PyArray_DIM(x_array, 0);
    int y_len = PyArray_DIM(y_array, 0);
    double* x_data = (double*)PyArray_DATA(x_array);
    double* y_data = (double*)PyArray_DATA(y_array);
    
    npy_intp dims[2] = {y_len, x_len};
    PyObject* X = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    PyObject* Y = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    
    double* X_data = (double*)PyArray_DATA((PyArrayObject*)X);
    double* Y_data = (double*)PyArray_DATA((PyArrayObject*)Y);
    
    for (int i = 0; i < y_len; i++) {
        for (int j = 0; j < x_len; j++) {
            X_data[i * x_len + j] = x_data[j];
            Y_data[i * x_len + j] = y_data[i];
        }
    }
    
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    
    return Py_BuildValue("OO", X, Y);
}

static PyObject* c_fft2(PyObject* self, PyObject* args) {
    PyObject* input_obj;
    if (!PyArg_ParseTuple(args, "O", &input_obj)) {
        return NULL;
    }
    
    PyArrayObject* input_array = (PyArrayObject*)PyArray_FROM_OTF(input_obj, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array == NULL) {
        return NULL;
    }
    
    int rows = PyArray_DIM(input_array, 0);
    int cols = PyArray_DIM(input_array, 1);
    
    npy_intp dims[2] = {rows, cols};
    PyObject* output = PyArray_ZEROS(2, dims, NPY_CDOUBLE, 0);
    
    double complex* input_data = (double complex*)PyArray_DATA(input_array);
    double complex* output_data = (double complex*)PyArray_DATA((PyArrayObject*)output);
    
    // Copy input to output
    memcpy(output_data, input_data, rows * cols * sizeof(double complex));
    
    // Perform FFT
    fft2d((Complex*)output_data, rows, cols, 0);
    
    Py_DECREF(input_array);
    return output;
}

static PyObject* c_ifft2(PyObject* self, PyObject* args) {
    PyObject* input_obj;
    if (!PyArg_ParseTuple(args, "O", &input_obj)) {
        return NULL;
    }
    
    PyArrayObject* input_array = (PyArrayObject*)PyArray_FROM_OTF(input_obj, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array == NULL) {
        return NULL;
    }
    
    int rows = PyArray_DIM(input_array, 0);
    int cols = PyArray_DIM(input_array, 1);
    
    npy_intp dims[2] = {rows, cols};
    PyObject* output = PyArray_ZEROS(2, dims, NPY_CDOUBLE, 0);
    
    double complex* input_data = (double complex*)PyArray_DATA(input_array);
    double complex* output_data = (double complex*)PyArray_DATA((PyArrayObject*)output);
    
    // Copy input to output
    memcpy(output_data, input_data, rows * cols * sizeof(double complex));
    
    // Perform IFFT
    fft2d((Complex*)output_data, rows, cols, 1);
    
    Py_DECREF(input_array);
    return output;
}

static PyObject* c_fftfreq(PyObject* self, PyObject* args) {
    int n;
    double d = 1.0;
    if (!PyArg_ParseTuple(args, "i|d", &n, &d)) {
        return NULL;
    }
    
    npy_intp dims[1] = {n};
    PyObject* array = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    double* data = (double*)PyArray_DATA((PyArrayObject*)array);
    
    for (int i = 0; i < n; i++) {
        if (i <= n / 2) {
            data[i] = (double)i / (n * d);
        } else {
            data[i] = (double)(i - n) / (n * d);
        }
    }
    
    return array;
}

static PyObject* c_abs_array(PyObject* self, PyObject* args) {
    PyObject* input_obj;
    if (!PyArg_ParseTuple(args, "O", &input_obj)) {
        return NULL;
    }
    
    PyArrayObject* input_array = (PyArrayObject*)PyArray_FROM_OTF(input_obj, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array == NULL) {
        return NULL;
    }
    
    int ndim = PyArray_NDIM(input_array);
    npy_intp* dims = PyArray_DIMS(input_array);
    
    PyObject* output = PyArray_ZEROS(ndim, dims, NPY_DOUBLE, 0);
    
    double complex* input_data = (double complex*)PyArray_DATA(input_array);
    double* output_data = (double*)PyArray_DATA((PyArrayObject*)output);
    
    int total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= dims[i];
    }
    
    for (int i = 0; i < total_size; i++) {
        output_data[i] = cabs(input_data[i]);
    }
    
    Py_DECREF(input_array);
    return output;
}

static PyObject* c_real_array(PyObject* self, PyObject* args) {
    PyObject* input_obj;
    if (!PyArg_ParseTuple(args, "O", &input_obj)) {
        return NULL;
    }
    
    PyArrayObject* input_array = (PyArrayObject*)PyArray_FROM_OTF(input_obj, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array == NULL) {
        return NULL;
    }
    
    int ndim = PyArray_NDIM(input_array);
    npy_intp* dims = PyArray_DIMS(input_array);
    
    PyObject* output = PyArray_ZEROS(ndim, dims, NPY_DOUBLE, 0);
    
    double complex* input_data = (double complex*)PyArray_DATA(input_array);
    double* output_data = (double*)PyArray_DATA((PyArrayObject*)output);
    
    int total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= dims[i];
    }
    
    for (int i = 0; i < total_size; i++) {
        output_data[i] = creal(input_data[i]);
    }
    
    Py_DECREF(input_array);
    return output;
}

// Method definitions
static PyMethodDef CBackendMethods[] = {
    {"zeros", c_zeros, METH_VARARGS, "Create zero array"},
    {"linspace", c_linspace, METH_VARARGS, "Create linearly spaced array"},
    {"meshgrid", c_meshgrid, METH_VARARGS, "Create coordinate arrays"},
    {"fft2", c_fft2, METH_VARARGS, "2D FFT"},
    {"ifft2", c_ifft2, METH_VARARGS, "2D IFFT"},
    {"fftfreq", c_fftfreq, METH_VARARGS, "FFT frequency array"},
    {"abs_array", c_abs_array, METH_VARARGS, "Absolute value of array"},
    {"real_array", c_real_array, METH_VARARGS, "Real part of array"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef cbackendmodule = {
    PyModuleDef_HEAD_INIT,
    "c_backend_core",
    "C backend core functions",
    -1,
    CBackendMethods
};

PyMODINIT_FUNC PyInit_c_backend_core(void) {
    import_array();
    return PyModule_Create(&cbackendmodule);
}