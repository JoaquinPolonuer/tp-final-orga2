# Simulación de Ondas 2D

Un proyecto de simulación de ondas 2D de alto rendimiento implementado con múltiples backends para comparación de performance.

## Características

- Simulación de ondas interactiva con visualización en tiempo real
- Múltiples implementaciones de backend:
  - Python (Python puro)
  - NumPy (operaciones vectorizadas)
  - C (extensión compilada)
  - C con AVX (optimizaciones SIMD)
  - Assembly (optimizaciones ASM)
- Herramientas de benchmarking para comparación de rendimiento
- Agregar fuentes de ondas haciendo click durante la simulación

## Uso

Primero compilar las extensiones:
```bash
make
```

Ejecutar la simulación interactiva:
```bash
python main.py
```

Correr benchmarks:
```bash
make benchmark
```

## Resultados

Los benchmarks de rendimiento y análisis están disponibles en el directorio `results/` y el informe detallado en `informe/`.