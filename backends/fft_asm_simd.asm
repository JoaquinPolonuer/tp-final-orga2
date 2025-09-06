; backends/fft_asm.asm  (SIMD-in-XMM: one complex per XMM)
; build: nasm -f elf64 backends/fft_asm.asm -o backends/fft_asm.o

global fft_1d_asm_simd

section .text

; ----------------------------------------------------------------------
; void bit_reverse_asm(Complex *x, int n)
; rdi = x, rsi = n
; ----------------------------------------------------------------------
bit_reverse_asm:
    ;--------------- PROLOGO ---------------
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8
    ;--------------- PROLOGO ---------------

    mov     rbx, rdi                ; rbx = base x
    mov     r12, rsi                ; r12 = n

    ; j = 0; i = 1;
    xor     r13d, r13d              ; r13 = j
    mov     r14, 1                  ; r14 = i

    ; for (i = 1; i < n; ++i)
.i_loop:
        cmp     r14, r12
        jge     .done

        ; bit = n >> 1;
        mov     r15, r12
        shr     r15, 1                  ; r15 = bit

    ; while (j & bit) { j ^= bit; bit >>= 1; }
.while_loop:
        test    r13, r15
        jz      .after_while
        xor     r13, r15                ; j ^= bit
        shr     r15, 1                  ; bit >>= 1
        jmp     .while_loop

.after_while:
        ; j ^= bit;
        xor     r13, r15

        ; if (i < j) swap(x[i], x[j])
        cmp     r14, r13
        jge     .no_swap

        ; addr_i = &x[i], addr_j = &x[j]   (Complex = 16 bytes)
        mov     rax, r14
        shl     rax, 4                   ; rax = i * 16
        mov     rdx, r13
        shl     rdx, 4                   ; rdx = j * 16

        ; temp = x[i] (swap 16 bytes at once)
        movupd  xmm0, [rbx + rax]
        movupd  xmm1, [rbx + rdx]
        movupd  [rbx + rax], xmm1
        movupd  [rbx + rdx], xmm0

.no_swap:
        inc     r14                      ; ++i
        jmp     .i_loop

.done:
    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret


; ----------------------------------------------------------------------
; void fft_1d_asm_simd *x, int n, int inverse)
; rdi = *x, rsi = n, rdx = inverse
; ----------------------------------------------------------------------
fft_1d_asm_simd
    ;--------------- PROLOGO ---------------
    push rbp
    mov  rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub  rsp, 8                      ; 8-byte scratch for x87<->SSE moves
    ;--------------- PROLOGO ---------------

    mov rbx, rdi                     ; rbx = *x
    mov r12, rsi                     ; r12 = n
    mov r13, rdx                     ; r13 = inverse

    ; In-place bit reversal
    call bit_reverse_asm

    mov     r14, 2                   ; len = 2
.out_loop:
        cmp     r14, r12             ; while (len <= n)
        jg      .inverse

        ; ------------------ w = cos(angle) + i sin(angle) ------------------
        ; angle = 2π/len * (inverse ? +1 : -1)
        fldpi                           ; ST0 = π
        fadd    st0, st0                ; ST0 = 2π
        mov     [rsp], r14              ; save len (int64)
        fild    qword [rsp]             ; ST0 = (double)len, ST1 = 2π
        fdivp   st1, st0                ; ST0 = 2π/len

        test    r13, r13
        jnz     .declarar_w
        fchs                               ; if (!inverse) angle = -angle

.declarar_w:
        fld     st0                         ; duplicate angle
        fsin                                ; ST0 = sin(angle), ST1 still angle
        fstp    qword [rsp]                 ; spill sin
        movsd   xmm7, [rsp]                 ; w_i in xmm7.low
        fcos                                ; ST0 = cos(angle)
        fstp    qword [rsp]                 ; spill cos
        movsd   xmm6, [rsp]                 ; w_r in xmm6.low
        ; pack w = {w_r, w_i} in xmm6
        unpcklpd xmm6, xmm7                 ; xmm6 = [w_r, w_i]
        ; -------------------------------------------------------------------

        ; -------- per-stage constants --------
        mov     r9, r14                     ; r9 = len
        shr     r9, 1                       ; r9 = len/2 (j max)
        mov     r11, r9
        shl     r11, 4                      ; r11 = (len/2) * 16 bytes
        ; -------------------------------------

        xor     r15, r15                    ; i = 0
    .mid_loop:
        ; ------- wn = 1 + 0i in xmm8 -------
        pxor    xmm9, xmm9                  ; 0.0
        fld1
        fstp    qword [rsp]
        movsd   xmm8, [rsp]                 ; xmm8.low = 1.0
        unpcklpd xmm8, xmm9                 ; xmm8 = [1.0, 0.0]
        ; ------------------------------------

        ; base del bloque i
        mov     rax, r15
        shl     rax, 4
        lea     r10, [rbx + rax]            ; r10 = &x[i]

        xor     rcx, rcx                    ; j = 0
    .in_loop:
        cmp     rcx, r9
        jge     .end_inner

        ; rdi = &x[i+j], rsi = &x[i+j+len/2]
        mov     rdx, rcx
        shl     rdx, 4                      ; j * 16
        lea     rdi, [r10 + rdx]
        lea     rsi, [rdi + r11]

        ; u = {u_r, u_i}, t = {t_r, t_i}
        movupd  xmm0, [rdi]
        movupd  xmm2, [rsi]

        ; ---------------- v = t * wn (complex) ----------------
        ; wr = [wn_r, wn_r], wi = [wn_i, wn_i]
        movapd  xmm4, xmm8
        shufpd  xmm4, xmm4, 0x0             ; wr
        movapd  xmm5, xmm8
        shufpd  xmm5, xmm5, 0x3             ; wi

        movapd  xmm11, xmm2
        mulpd   xmm11, xmm4                 ; a = t * wr = [tr*wr, ti*wr]
        movapd  xmm12, xmm2
        shufpd  xmm12, xmm12, 0x1           ; [ti, tr]
        mulpd   xmm12, xmm5                 ; b = swap(t) * wi = [ti*wi, tr*wi]
        addsubpd xmm11, xmm12               ; v = [a0-b0, a1+b1]
        ; ------------------------------------------------------

        ; x[i+j] = u + v
        movapd  xmm13, xmm0
        addpd   xmm13, xmm11
        movupd  [rdi], xmm13

        ; x[i+j+len/2] = u - v
        subpd   xmm0, xmm11
        movupd  [rsi], xmm0

        ; --------------- wn = wn * w ----------------
        ; same complex multiply pattern, wn in xmm8, w in xmm6
        movapd  xmm4, xmm6
        shufpd  xmm4, xmm4, 0x0             ; wr
        movapd  xmm5, xmm6
        shufpd  xmm5, xmm5, 0x3             ; wi

        movapd  xmm13, xmm8
        mulpd   xmm13, xmm4                 ; a = wn * wr
        movapd  xmm14, xmm8
        shufpd  xmm14, xmm14, 0x1           ; swap(wn)
        mulpd   xmm14, xmm5                 ; b = swap(wn) * wi
        addsubpd xmm13, xmm14               ; new wn
        movapd  xmm8, xmm13
        ; -------------------------------------------

        ; j++
        inc     rcx
        jmp     .in_loop

    .end_inner:
        add     r15, r14                    ; i += len
        cmp     r15, r12
        jl      .mid_loop

        shl     r14, 1                      ; len <<= 1
        jmp     .out_loop

    ; --------------------------- inverse scaling ---------------------------
    .inverse:
        ; if (!inverse) skip 1/n scaling
        test    r13, r13
        jz      .end_fft

        ; xmm10 = [n, n]
        cvtsi2sd xmm10, r12
        shufpd   xmm10, xmm10, 0x0

        ; i = 0
        xor     r15, r15
    .inverse_loop:
        cmp     r15, r12
        jge     .end_fft

        mov     rax, r15
        shl     rax, 4                      ; i * 16
        movupd  xmm0, [rbx + rax]           ; {real, imag}
        divpd   xmm0, xmm10
        movupd  [rbx + rax], xmm0

        inc     r15
        jmp     .inverse_loop

    .end_fft:
        add rsp, 8
        pop r15
        pop r14
        pop r13
        pop r12
        pop rbx
        pop rbp
        ret