section .text
global asm_complex_add
global asm_complex_sub
global asm_complex_mul

; Complex structure:
; - 8 bytes: real (double)
; - 8 bytes: imag (double)
; Total: 16 bytes
;
; x86-64 calling convention for pointers:
; - Complex *a: pointer in rdi
; - Complex *b: pointer in rsi
; - Complex *result: pointer in rdx

; void asm_complex_add(Complex *a, Complex *b, Complex *result)
; Parameters: a=rdi, b=rsi, result=rdx
asm_complex_add:
    movupd  xmm0, [rdi]       ; xmm0 = [a.real, a.imag]
    addpd   xmm0, [rsi]       ; xmm0 += [b.real, b.imag]
    movupd  [rdx], xmm0       ; store result
    ret

; void asm_complex_sub(Complex *a, Complex *b, Complex *result)
; Parameters: a=rdi, b=rsi, result=rdx
asm_complex_sub:
    movupd  xmm0, [rdi]       ; xmm0 = [a.real, a.imag]
    subpd   xmm0, [rsi]       ; xmm0 -= [b.real, b.imag]
    movupd  [rdx], xmm0
    ret

; void asm_complex_mul(Complex *a, Complex *b, Complex *result)
; a=rdi, b=rsi, result=rdx
asm_complex_mul:
    movupd  xmm0, [rdi]        ; xmm0 = [ar, ai]
    movupd  xmm1, [rsi]        ; xmm1 = [br, bi]

    movapd  xmm2, xmm1         ; xmm2 = [br, bi]
    movapd  xmm3, xmm0         ; xmm3 = [ar, ai]

    ; Duplicate br and bi into lanes
    unpcklpd xmm1, xmm1        ; xmm1 = [br, br]
    unpckhpd xmm2, xmm2        ; xmm2 = [bi, bi]

    ; Cross products
    mulpd   xmm0, xmm1         ; xmm0 = [ar*br, ai*br]
    mulpd   xmm3, xmm2         ; xmm3 = [ar*bi, ai*bi]

    ; Arrange for add/sub: want [ai*bi, ar*bi] to pair with [ar*br, ai*br]
    shufpd  xmm3, xmm3, 0x1   ; xmm3 = [ai*bi, ar*bi]

    ; (real, imag) = (ar*br - ai*bi, ai*br + ar*bi)
    addsubpd xmm0, xmm3        ; xmm0 = [real, imag]

    movupd  [rdx], xmm0
    ret