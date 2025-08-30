# Wave Simulation Performance Comparison Project
# Makefile for building different backends

.PHONY: all clean build-c benchmark

# Default target
all: build-c

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -f backends/*.so
	rm -f backends/*.dylib
	rm -f backends/*.dll
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Clean complete!"

# Build C backend
build-c:
	@echo "Building C backend..."
	$(MAKE) clean
	python setup.py build_ext --inplace
	@echo "C backend built successfully!"

# Run benchmark comparison
benchmark: build-c
	@echo "Running performance benchmark..."
	python benchmark.py
