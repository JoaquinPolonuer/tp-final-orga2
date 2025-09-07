#!/usr/bin/env python3

import sys
import math
from backends import c_backend_asm

def complex_mul_reference(a_real, a_imag, b_real, b_imag):
    """Reference implementation of complex multiplication"""
    real = a_real * b_real - a_imag * b_imag
    imag = a_real * b_imag + a_imag * b_real
    return (real, imag)

def test_complex_multiplication():
    """Test the COMPLEX_MUL assembly macro against reference implementation"""
    
    # Test cases: (a_real, a_imag, b_real, b_imag, description)
    test_cases = [
        # Basic cases
        (1.0, 0.0, 1.0, 0.0, "Real numbers: 1 * 1"),
        (2.0, 0.0, 3.0, 0.0, "Real numbers: 2 * 3"),
        (0.0, 1.0, 0.0, 1.0, "Pure imaginary: i * i"),
        (1.0, 1.0, 1.0, 1.0, "Complex: (1+i) * (1+i)"),
        
        # Edge cases
        (0.0, 0.0, 1.0, 1.0, "Zero multiplication"),
        (1.0, 0.0, 0.0, 1.0, "Real * pure imaginary"),
        
        # More complex cases
        (2.0, 3.0, 4.0, 5.0, "General case: (2+3i) * (4+5i)"),
        (-1.0, 2.0, 3.0, -4.0, "Negative values: (-1+2i) * (3-4i)"),
        
        # Unity and special values
        (math.cos(math.pi/4), math.sin(math.pi/4), math.cos(math.pi/4), math.sin(math.pi/4), "e^(iπ/4) * e^(iπ/4)"),
        (math.sqrt(2), math.sqrt(3), math.sqrt(5), math.sqrt(7), "Irrational numbers"),
        
        # Large and small numbers
        (1e6, 1e6, 1e-6, 1e-6, "Large and small numbers"),
        (1e-15, 1e-15, 1e15, 1e15, "Very small and very large")
    ]
    
    print("Testing COMPLEX_MUL assembly macro")
    print("=" * 50)
    
    all_passed = True
    tolerance = 1e-14  # Tolerance for floating point comparison
    
    for i, (a_real, a_imag, b_real, b_imag, description) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {description}")
        print(f"  Input: ({a_real:+.6f}{a_imag:+.6f}i) * ({b_real:+.6f}{b_imag:+.6f}i)")
        
        # Get results from both implementations
        expected = complex_mul_reference(a_real, a_imag, b_real, b_imag)
        actual = c_backend_asm.test_complex_mul(a_real, a_imag, b_real, b_imag)
        
        # Check if results match within tolerance
        real_error = abs(expected[0] - actual[0])
        imag_error = abs(expected[1] - actual[1])
        
        print(f"  Expected: {expected[0]:+.6f}{expected[1]:+.6f}i")
        print(f"  Actual:   {actual[0]:+.6f}{actual[1]:+.6f}i")
        print(f"  Error:    {real_error:.2e} + {imag_error:.2e}i")
        
        if real_error <= tolerance and imag_error <= tolerance:
            print("  ✓ PASSED")
        else:
            print("  ✗ FAILED")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests PASSED! The COMPLEX_MUL macro works correctly.")
    else:
        print("❌ Some tests FAILED. Check the macro implementation.")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = test_complex_multiplication()
        sys.exit(0 if success else 1)
    except ImportError as e:
        print(f"Error importing backend: {e}")
        print("Make sure to build the C extension first with: python setup.py build_ext --inplace")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)