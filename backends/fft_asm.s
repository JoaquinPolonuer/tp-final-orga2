.section .text
.global fft_1d_asm
.global bit_reverse_asm

# Complex number structure: real (8 bytes) + imag (8 bytes) = 16 bytes
# Function: fft_1d_asm(Complex *x, int n, int inverse)
# Parameters:
#   rdi: Complex *x (array pointer)
#   rsi: n (array size)
#   rdx: inverse (0 for forward, non-zero for inverse)

fft_1d_asm:
    push %rbp
    mov %rsp, %rbp
    push %rbx
    push %r12
    push %r13
    push %r14
    push %r15
    
    # Save parameters
    mov %rdi, %r12    # x array pointer
    mov %rsi, %r13    # n
    mov %rdx, %r14    # inverse flag
    
    # Check precondition: n > 0 and n is power of 2
    test %r13, %r13
    jz .error
    mov %r13, %rax
    dec %rax
    and %r13, %rax
    jnz .error
    
    # Call bit reverse
    mov %r12, %rdi
    mov %r13, %rsi
    call bit_reverse_asm
    
    # Main FFT loop: for (len = 2; len <= n; len <<= 1)
    mov $2, %r15      # len = 2
    
.outer_loop:
    cmp %r13, %r15    # if len > n, exit
    jg .normalize
    
    # Calculate angle = 2π/len * (inverse ? 1 : -1)
    cvtsi2sd %r15, %xmm0    # convert len to double
    movsd .two_pi(%rip), %xmm1
    divsd %xmm0, %xmm1      # 2π/len
    
    test %r14, %r14         # check inverse flag
    jz .forward_angle
    movsd %xmm1, %xmm0      # positive angle for inverse
    jmp .calc_w
.forward_angle:
    xorpd %xmm0, %xmm0
    subsd %xmm1, %xmm0      # negative angle for forward
    
.calc_w:
    # Calculate w = cos(angle) + i*sin(angle)
    movsd %xmm0, %xmm2      # save angle
    call cos@PLT            # cos(angle) in xmm0
    movsd %xmm0, -16(%rbp)  # save cos(angle)
    movsd %xmm2, %xmm0      # restore angle
    call sin@PLT            # sin(angle) in xmm0
    movsd %xmm0, -8(%rbp)   # save sin(angle)
    
    # Inner loop: for (i = 0; i < n; i += len)
    xor %rbx, %rbx          # i = 0
    
.middle_loop:
    cmp %r13, %rbx          # if i >= n, continue outer loop
    jge .next_len
    
    # wn = 1.0 + 0.0i
    movsd .one(%rip), %xmm0
    movsd .zero(%rip), %xmm1
    movsd %xmm0, -32(%rbp)  # wn.real = 1.0
    movsd %xmm1, -24(%rbp)  # wn.imag = 0.0
    
    # Inner loop: for (j = 0; j < len/2; j++)
    xor %rcx, %rcx          # j = 0
    mov %r15, %rax
    shr $1, %rax            # len/2
    
.inner_loop:
    cmp %rax, %rcx          # if j >= len/2, continue middle loop
    jge .next_i
    
    # Calculate indices
    mov %rbx, %r8           # i
    add %rcx, %r8           # i + j
    mov %rbx, %r9           # i
    add %rcx, %r9           # i + j
    mov %r15, %r10
    shr $1, %r10            # len/2
    add %r10, %r9           # i + j + len/2
    
    # Load u = x[i + j]
    mov %r8, %r10
    shl $4, %r10            # multiply by 16 (sizeof Complex)
    add %r12, %r10          # address of x[i + j]
    movsd (%r10), %xmm0     # u.real
    movsd 8(%r10), %xmm1    # u.imag
    
    # Load x[i + j + len/2]
    mov %r9, %r11
    shl $4, %r11            # multiply by 16
    add %r12, %r11          # address of x[i + j + len/2]
    movsd (%r11), %xmm2     # x[i + j + len/2].real
    movsd 8(%r11), %xmm3    # x[i + j + len/2].imag
    
    # Load wn
    movsd -32(%rbp), %xmm4  # wn.real
    movsd -24(%rbp), %xmm5  # wn.imag
    
    # Complex multiplication: v = x[i + j + len/2] * wn
    # v.real = x.real * wn.real - x.imag * wn.imag
    # v.imag = x.real * wn.imag + x.imag * wn.real
    movsd %xmm2, %xmm6      # copy x.real
    mulsd %xmm4, %xmm6      # x.real * wn.real
    movsd %xmm3, %xmm7      # copy x.imag
    mulsd %xmm5, %xmm7      # x.imag * wn.imag
    subsd %xmm7, %xmm6      # v.real = x.real * wn.real - x.imag * wn.imag
    
    movsd %xmm2, %xmm7      # copy x.real
    mulsd %xmm5, %xmm7      # x.real * wn.imag
    movsd %xmm3, %xmm8      # copy x.imag
    mulsd %xmm4, %xmm8      # x.imag * wn.real
    addsd %xmm8, %xmm7      # v.imag = x.real * wn.imag + x.imag * wn.real
    
    # x[i + j] = u + v
    addsd %xmm6, %xmm0      # u.real + v.real
    addsd %xmm7, %xmm1      # u.imag + v.imag
    movsd %xmm0, (%r10)     # store x[i + j].real
    movsd %xmm1, 8(%r10)    # store x[i + j].imag
    
    # x[i + j + len/2] = u - v
    movsd (%r10), %xmm0     # reload u.real (now u + v)
    movsd 8(%r10), %xmm1    # reload u.imag (now u + v)
    subsd %xmm6, %xmm0      # (u + v) - v = u - v (real)
    subsd %xmm7, %xmm1      # (u + v) - v = u - v (imag)
    subsd %xmm6, %xmm0      # u.real - v.real
    subsd %xmm7, %xmm1      # u.imag - v.imag
    movsd %xmm0, (%r11)     # store x[i + j + len/2].real
    movsd %xmm1, 8(%r11)    # store x[i + j + len/2].imag
    
    # wn *= w (complex multiplication)
    movsd -32(%rbp), %xmm0  # wn.real
    movsd -24(%rbp), %xmm1  # wn.imag
    movsd -16(%rbp), %xmm2  # w.real
    movsd -8(%rbp), %xmm3   # w.imag
    
    # new_wn.real = wn.real * w.real - wn.imag * w.imag
    movsd %xmm0, %xmm4
    mulsd %xmm2, %xmm4      # wn.real * w.real
    movsd %xmm1, %xmm5
    mulsd %xmm3, %xmm5      # wn.imag * w.imag
    subsd %xmm5, %xmm4      # new_wn.real
    
    # new_wn.imag = wn.real * w.imag + wn.imag * w.real
    mulsd %xmm3, %xmm0      # wn.real * w.imag
    mulsd %xmm2, %xmm1      # wn.imag * w.real
    addsd %xmm1, %xmm0      # new_wn.imag
    
    movsd %xmm4, -32(%rbp)  # save new wn.real
    movsd %xmm0, -24(%rbp)  # save new wn.imag
    
    inc %rcx                # j++
    jmp .inner_loop
    
.next_i:
    add %r15, %rbx          # i += len
    jmp .middle_loop
    
.next_len:
    shl $1, %r15            # len <<= 1
    jmp .outer_loop
    
.normalize:
    # If inverse, normalize by dividing by n
    test %r14, %r14
    jz .done
    
    cvtsi2sd %r13, %xmm0    # convert n to double
    xor %rbx, %rbx          # i = 0
    
.normalize_loop:
    cmp %r13, %rbx
    jge .done
    
    mov %rbx, %rax
    shl $4, %rax            # multiply by 16
    add %r12, %rax          # address of x[i]
    
    movsd (%rax), %xmm1     # x[i].real
    movsd 8(%rax), %xmm2    # x[i].imag
    divsd %xmm0, %xmm1      # x[i].real / n
    divsd %xmm0, %xmm2      # x[i].imag / n
    movsd %xmm1, (%rax)     # store normalized real
    movsd %xmm2, 8(%rax)    # store normalized imag
    
    inc %rbx
    jmp .normalize_loop
    
.done:
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbx
    mov %rbp, %rsp
    pop %rbp
    ret

.error:
    # For now, just return - in practice you'd want proper error handling
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbx
    mov %rbp, %rsp
    pop %rbp
    ret

# Bit reversal function: bit_reverse_asm(Complex *x, int n)
bit_reverse_asm:
    push %rbp
    mov %rsp, %rbp
    push %rbx
    push %r12
    push %r13
    push %r14
    
    mov %rdi, %r12    # x array pointer
    mov %rsi, %r13    # n
    
    xor %rbx, %rbx    # j = 0
    mov $1, %rcx      # i = 1
    
.bit_reverse_loop:
    cmp %r13, %rcx    # if i >= n, exit
    jge .bit_reverse_done
    
    mov %r13, %rax    # bit = n >> 1
    shr $1, %rax
    
.bit_reverse_inner:
    test %rbx, %rax   # while (j & bit)
    jz .bit_reverse_update
    xor %rax, %rbx    # j ^= bit
    shr $1, %rax      # bit >>= 1
    jmp .bit_reverse_inner
    
.bit_reverse_update:
    xor %rax, %rbx    # j ^= bit
    
    cmp %rbx, %rcx    # if (i < j)
    jge .bit_reverse_next
    
    # Swap x[i] and x[j]
    mov %rcx, %rax
    shl $4, %rax      # i * 16
    add %r12, %rax    # address of x[i]
    
    mov %rbx, %rdx
    shl $4, %rdx      # j * 16
    add %r12, %rdx    # address of x[j]
    
    # Load x[i]
    movsd (%rax), %xmm0     # x[i].real
    movsd 8(%rax), %xmm1    # x[i].imag
    
    # Load x[j]
    movsd (%rdx), %xmm2     # x[j].real
    movsd 8(%rdx), %xmm3    # x[j].imag
    
    # Store x[j] at x[i]
    movsd %xmm2, (%rax)
    movsd %xmm3, 8(%rax)
    
    # Store x[i] at x[j]
    movsd %xmm0, (%rdx)
    movsd %xmm1, 8(%rdx)
    
.bit_reverse_next:
    inc %rcx          # i++
    jmp .bit_reverse_loop
    
.bit_reverse_done:
    pop %r14
    pop %r13
    pop %r12
    pop %rbx
    mov %rbp, %rsp
    pop %rbp
    ret

.section .rodata
.align 8
.two_pi:
    .double 6.283185307179586
.one:
    .double 1.0
.zero:
    .double 0.0