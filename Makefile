.PHONY: all clean build-c benchmark

# Default
all: build-c

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -f backends/*.so
	rm -f backends/*.dylib
	rm -f backends/*.dll
	rm -f backends/*.o
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Clean complete!"

build-c:
	@echo "Building C backend..."
	$(MAKE) clean
	uv run python setup.py build_ext --inplace
	@echo "C backend built successfully!"

benchmark: build-c
	@echo "Running performance benchmark..."
	uv run python benchmark.py
