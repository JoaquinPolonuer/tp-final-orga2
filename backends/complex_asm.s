.section .text
.global asm_complex_add
.global asm_complex_sub
.global asm_complex_mul

# Complex structure:
# - 8 bytes: real (double)
# - 8 bytes: imag (double)
# Total: 16 bytes
#
# x86-64 calling convention for structs:
# - Complex a: a.real in xmm0, a.imag in xmm1
# - Complex b: b.real in xmm2, b.imag in xmm3
# - Return: result.real in xmm0, result.imag in xmm1

# Complex asm_complex_add(Complex a, Complex b)
# Parameters: a.real=xmm0, a.imag=xmm1, b.real=xmm2, b.imag=xmm3
# Returns: result.real=xmm0, result.imag=xmm1
asm_complex_add:
    # Add real parts: xmm0 = a.real + b.real
    addsd %xmm2, %xmm0
    
    # Add imaginary parts: xmm1 = a.imag + b.imag
    addsd %xmm3, %xmm1
    
    ret

# Complex asm_complex_sub(Complex a, Complex b)
# Parameters: a.real=xmm0, a.imag=xmm1, b.real=xmm2, b.imag=xmm3
# Returns: result.real=xmm0, result.imag=xmm1
asm_complex_sub:
    # Subtract real parts: xmm0 = a.real - b.real
    subsd %xmm2, %xmm0
    
    # Subtract imaginary parts: xmm1 = a.imag - b.imag
    subsd %xmm3, %xmm1
    
    ret

# Complex asm_complex_mul(Complex a, Complex b)
# Parameters: a.real=xmm0, a.imag=xmm1, b.real=xmm2, b.imag=xmm3
# Returns: result.real=xmm0, result.imag=xmm1
# Formula: (a.real + i*a.imag) * (b.real + i*b.imag)
#        = (a.real*b.real - a.imag*b.imag) + i*(a.real*b.imag + a.imag*b.real)
asm_complex_mul:
    # Save original values we'll need later
    movsd %xmm0, %xmm4       # xmm4 = a.real
    movsd %xmm1, %xmm5       # xmm5 = a.imag
    
    # Calculate real part: a.real*b.real - a.imag*b.imag
    mulsd %xmm2, %xmm0       # xmm0 = a.real * b.real
    mulsd %xmm3, %xmm5       # xmm5 = a.imag * b.imag
    subsd %xmm5, %xmm0       # xmm0 = a.real*b.real - a.imag*b.imag (result.real)
    
    # Calculate imaginary part: a.real*b.imag + a.imag*b.real
    mulsd %xmm3, %xmm4       # xmm4 = a.real * b.imag
    mulsd %xmm2, %xmm1       # xmm1 = a.imag * b.real
    addsd %xmm4, %xmm1       # xmm1 = a.real*b.imag + a.imag*b.real (result.imag)
    
    ret