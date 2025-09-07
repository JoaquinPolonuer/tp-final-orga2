#ifndef WAVE_SIMULATION_COMMON_H
#define WAVE_SIMULATION_COMMON_H

#include <Python.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

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

// Common helper functions
Complex complex_add(Complex a, Complex b);
Complex complex_mul(Complex a, Complex b);

// Wave simulation functions
WaveSimulation *create_wave_simulation(int size, double domain_size, double wave_speed, double dt);
void wave_sim_add_source(WaveSimulation *sim, double x_pos, double y_pos,
                         double amplitude, double frequency, double width);
void wave_sim_step(WaveSimulation *sim);
PyObject *wave_sim_get_intensity(WaveSimulation *sim);
PyObject *wave_sim_get_real_part(WaveSimulation *sim);

// Python interface functions
PyObject *c_create_simulation(PyObject *self, PyObject *args);
PyObject *c_add_wave_source(PyObject *self, PyObject *args);
PyObject *c_step_simulation(PyObject *self, PyObject *args);
PyObject *c_get_intensity(PyObject *self, PyObject *args);
PyObject *c_get_real_part(PyObject *self, PyObject *args);

// FFT function pointer - each backend will provide its own implementation
extern void (*fft2d)(Complex *data, int rows, int cols, int inverse);

#endif // WAVE_SIMULATION_COMMON_H