#include "wave_simulation_common.h"

// Puntero a funcion de FFT, cada backend setea la propia
void (*fft2d)(Complex *data, int rows, int cols, int inverse) = NULL;

Complex complex_add(Complex a, Complex b)
{
    Complex result = {a.real + b.real, a.imag + b.imag};
    return result;
}

Complex complex_sub(Complex a, Complex b)
{
    Complex result = {a.real - b.real, a.imag - b.imag};
    return result;
}

Complex complex_mul(Complex a, Complex b)
{
    Complex result = {
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real};
    return result;
}

void bit_reverse(Complex *x, int n)
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

WaveSimulation *create_wave_simulation(int size, double domain_size, double wave_speed, double dt)
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

void wave_sim_add_source(WaveSimulation *sim, double x_pos, double y_pos,
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

void wave_sim_step(WaveSimulation *sim)
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

// ------------------------ Interfaz con Python ------------------------------

PyObject *wave_sim_get_intensity(WaveSimulation *sim)
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

PyObject *wave_sim_get_real_part(WaveSimulation *sim)
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

PyObject *c_create_simulation(PyObject *self, PyObject *args)
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

PyObject *c_add_wave_source(PyObject *self, PyObject *args)
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

PyObject *c_step_simulation(PyObject *self, PyObject *args)
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

PyObject *c_get_intensity(PyObject *self, PyObject *args)
{
    PyObject *ptr_obj;
    if (!PyArg_ParseTuple(args, "O", &ptr_obj))
    {
        return NULL;
    }

    WaveSimulation *sim = (WaveSimulation *)PyLong_AsVoidPtr(ptr_obj);
    return wave_sim_get_intensity(sim);
}

PyObject *c_get_real_part(PyObject *self, PyObject *args)
{
    PyObject *ptr_obj;
    if (!PyArg_ParseTuple(args, "O", &ptr_obj))
    {
        return NULL;
    }

    WaveSimulation *sim = (WaveSimulation *)PyLong_AsVoidPtr(ptr_obj);
    return wave_sim_get_real_part(sim);
}

PyMethodDef BackendMethods[] = {
    {"create_simulation", c_create_simulation, METH_VARARGS, "Create wave simulation"},
    {"add_wave_source", c_add_wave_source, METH_VARARGS, "Add wave source"},
    {"step_simulation", c_step_simulation, METH_VARARGS, "Step simulation"},
    {"get_intensity", c_get_intensity, METH_VARARGS, "Get wave intensity"},
    {"get_real_part", c_get_real_part, METH_VARARGS, "Get real part of wave"},
    {NULL, NULL, 0, NULL}};

PyObject *create_python_module(const char *module_name, const char *module_doc)
{
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        NULL,
        NULL,
        -1,
        BackendMethods};

    moduledef.m_name = module_name;
    moduledef.m_doc = module_doc;

    return PyModule_Create(&moduledef);
}