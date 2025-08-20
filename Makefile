# Wave Simulation Performance Comparison Project
# Makefile for building different backends

.PHONY: all clean build-c test-all benchmark help install-deps

# Default target
all: build-c

# Build C backend
build-c:
	@echo "Building C backend..."
	python setup.py build_ext --inplace
	@echo "C backend built successfully!"

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

# Install Python dependencies
install-deps:
	@echo "Installing Python dependencies..."
	pip install numpy matplotlib setuptools

# Test all backends
test-all: build-c
	@echo "Testing NumPy backend..."
	python -c "from main import run_simulation; import matplotlib; matplotlib.use('Agg'); run_simulation('numpy')"
	@echo "Testing Pure Python backend..."
	python -c "from main import run_simulation; import matplotlib; matplotlib.use('Agg'); run_simulation('python')"
	@echo "Testing C backend..."
	python -c "from main import run_simulation; import matplotlib; matplotlib.use('Agg'); run_simulation('c')"
	@echo "All backends tested successfully!"

# Run benchmark comparison
benchmark: build-c
	@echo "Running performance benchmark..."
	python benchmark.py

# Run with specific backend
run-numpy:
	python -c "import main; main.backend_choice = 'numpy'; exec(open('main.py').read())"

run-python:
	python -c "import main; main.backend_choice = 'python'; exec(open('main.py').read())"

run-c: build-c
	python -c "import main; main.backend_choice = 'c'; exec(open('main.py').read())"

# Development targets
dev-build: clean build-c

# Force rebuild
rebuild: clean build-c

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build all backends (default)"
	@echo "  build-c      - Build C backend only"
	@echo "  clean        - Remove build artifacts"
	@echo "  install-deps - Install Python dependencies"
	@echo "  test-all     - Test all backends"
	@echo "  benchmark    - Run performance benchmark"
	@echo "  run-numpy    - Run simulation with NumPy backend"
	@echo "  run-python   - Run simulation with Pure Python backend"
	@echo "  run-c        - Run simulation with C backend"
	@echo "  dev-build    - Clean and rebuild for development"
	@echo "  rebuild      - Force clean rebuild"
	@echo "  help         - Show this help message"