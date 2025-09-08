# Simulación de Ondas 2D

Un proyecto de simulación de ondas 2D de alto rendimiento implementado con múltiples backends para comparación de performance.

![Simulación](informe/extra/live_visualization.png)

## Características

- Simulación de ondas interactiva con visualización en tiempo real
- Múltiples implementaciones de backend:
  - Python (Python puro)
  - NumPy (operaciones vectorizadas)
  - C
  - C con AVX (optimizaciones SIMD)
  - Assembly (optimizaciones ASM)
- Benchmarking para comparación de rendimiento

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