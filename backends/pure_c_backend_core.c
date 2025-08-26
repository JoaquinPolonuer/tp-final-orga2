#include <Python.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct
{
    double real;
    double imag;
} Complex;

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

static Complex complex_exp(Complex z)
{
    double exp_real = exp(z.real);
    Complex result = {
        exp_real * cos(z.imag),
        exp_real * sin(z.imag)};
    return result;
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

// 1D FFT implementation (Cooley-Tukey)
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

// Python wrapper functions
static PyObject *pure_c_zeros(PyObject *self, PyObject *args)
{
    int rows, cols;
    if (!PyArg_ParseTuple(args, "ii", &rows, &cols))
    {
        return NULL;
    }

    PyObject *py_list = PyList_New(rows);
    for (int i = 0; i < rows; i++)
    {
        PyObject *row = PyList_New(cols);
        for (int j = 0; j < cols; j++)
        {
            PyObject *zero = PyComplex_FromDoubles(0.0, 0.0);
            PyList_SetItem(row, j, zero);
        }
        PyList_SetItem(py_list, i, row);
    }

    return py_list;
}

static PyObject *pure_c_linspace(PyObject *self, PyObject *args)
{
    double start, stop;
    int num;
    if (!PyArg_ParseTuple(args, "ddi", &start, &stop, &num))
    {
        return NULL;
    }

    PyObject *py_list = PyList_New(num);

    if (num == 1)
    {
        PyList_SetItem(py_list, 0, PyFloat_FromDouble(start));
    }
    else
    {
        double step = (stop - start) / (num - 1);
        for (int i = 0; i < num; i++)
        {
            double val = start + i * step;
            PyList_SetItem(py_list, i, PyFloat_FromDouble(val));
        }
    }

    return py_list;
}

static PyObject *pure_c_meshgrid(PyObject *self, PyObject *args)
{
    PyObject *x_obj, *y_obj;
    if (!PyArg_ParseTuple(args, "OO", &x_obj, &y_obj))
    {
        return NULL;
    }

    if (!PyList_Check(x_obj) || !PyList_Check(y_obj))
    {
        PyErr_SetString(PyExc_TypeError, "Expected lists");
        return NULL;
    }

    int x_len = (int)PyList_Size(x_obj);
    int y_len = (int)PyList_Size(y_obj);

    PyObject *X = PyList_New(y_len);
    PyObject *Y = PyList_New(y_len);

    for (int i = 0; i < y_len; i++)
    {
        PyObject *X_row = PyList_New(x_len);
        PyObject *Y_row = PyList_New(x_len);

        for (int j = 0; j < x_len; j++)
        {
            PyObject *x_val = PyList_GetItem(x_obj, j);
            PyObject *y_val = PyList_GetItem(y_obj, i);

            Py_INCREF(x_val);
            Py_INCREF(y_val);

            PyList_SetItem(X_row, j, x_val);
            PyList_SetItem(Y_row, j, y_val);
        }

        PyList_SetItem(X, i, X_row);
        PyList_SetItem(Y, i, Y_row);
    }

    return Py_BuildValue("OO", X, Y);
}

static PyObject *pure_c_fft2(PyObject *self, PyObject *args)
{
    PyObject *input_obj;
    if (!PyArg_ParseTuple(args, "O", &input_obj))
    {
        return NULL;
    }

    clock_t start_total = clock();
    
    clock_t start_conversion = clock();
    int rows, cols;
    Complex *data = python_to_c_array(input_obj, &rows, &cols);
    if (data == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Expected list of lists");
        return NULL;
    }
    clock_t end_conversion = clock();
    double python_to_c_time = ((double)(end_conversion - start_conversion)) / CLOCKS_PER_SEC;

    clock_t start_fft = clock();
    fft2d(data, rows, cols, 0);
    clock_t end_fft = clock();
    double fft_time = ((double)(end_fft - start_fft)) / CLOCKS_PER_SEC;

    clock_t start_back_conversion = clock();
    PyObject *result = c_to_python_array(data, rows, cols);
    clock_t end_back_conversion = clock();
    double c_to_python_time = ((double)(end_back_conversion - start_back_conversion)) / CLOCKS_PER_SEC;
    
    clock_t end_total = clock();
    double total_time = ((double)(end_total - start_total)) / CLOCKS_PER_SEC;

    printf("FFT2D Timing:\n");
    printf("  Python->C conversion: %.6f seconds\n", python_to_c_time);
    printf("  FFT computation:      %.6f seconds\n", fft_time);
    printf("  C->Python conversion: %.6f seconds\n", c_to_python_time);
    printf("  Total time:           %.6f seconds\n", total_time);

    free(data);

    return result;
}

static PyObject *pure_c_ifft2(PyObject *self, PyObject *args)
{
    PyObject *input_obj;
    if (!PyArg_ParseTuple(args, "O", &input_obj))
    {
        return NULL;
    }

    clock_t start_total = clock();
    
    clock_t start_conversion = clock();
    int rows, cols;
    Complex *data = python_to_c_array(input_obj, &rows, &cols);
    if (data == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Expected list of lists");
        return NULL;
    }
    clock_t end_conversion = clock();
    double python_to_c_time = ((double)(end_conversion - start_conversion)) / CLOCKS_PER_SEC;

    clock_t start_ifft = clock();
    fft2d(data, rows, cols, 1);
    clock_t end_ifft = clock();
    double ifft_time = ((double)(end_ifft - start_ifft)) / CLOCKS_PER_SEC;

    clock_t start_back_conversion = clock();
    PyObject *result = c_to_python_array(data, rows, cols);
    clock_t end_back_conversion = clock();
    double c_to_python_time = ((double)(end_back_conversion - start_back_conversion)) / CLOCKS_PER_SEC;
    
    clock_t end_total = clock();
    double total_time = ((double)(end_total - start_total)) / CLOCKS_PER_SEC;

    printf("IFFT2D Timing:\n");
    printf("  Python->C conversion: %.6f seconds\n", python_to_c_time);
    printf("  IFFT computation:     %.6f seconds\n", ifft_time);
    printf("  C->Python conversion: %.6f seconds\n", c_to_python_time);
    printf("  Total time:           %.6f seconds\n", total_time);

    free(data);

    return result;
}

static PyObject *pure_c_fftfreq(PyObject *self, PyObject *args)
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

static PyObject *pure_c_abs_array(PyObject *self, PyObject *args)
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

static PyObject *pure_c_real_array(PyObject *self, PyObject *args)
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
    {"zeros", pure_c_zeros, METH_VARARGS, "Create zero array"},
    {"linspace", pure_c_linspace, METH_VARARGS, "Create linearly spaced array"},
    {"meshgrid", pure_c_meshgrid, METH_VARARGS, "Create coordinate arrays"},
    {"fft2", pure_c_fft2, METH_VARARGS, "2D FFT"},
    {"ifft2", pure_c_ifft2, METH_VARARGS, "2D IFFT"},
    {"fftfreq", pure_c_fftfreq, METH_VARARGS, "FFT frequency array"},
    {"abs_array", pure_c_abs_array, METH_VARARGS, "Absolute value of array"},
    {"real_array", pure_c_real_array, METH_VARARGS, "Real part of array"},
    {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef purecbackendmodule = {
    PyModuleDef_HEAD_INIT,
    "pure_c_backend_core",
    "Pure C backend core functions without NumPy",
    -1,
    PureCBackendMethods};

PyMODINIT_FUNC PyInit_pure_c_backend_core(void)
{
    return PyModule_Create(&purecbackendmodule);
}