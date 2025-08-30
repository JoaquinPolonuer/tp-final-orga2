; fft_asm_avx.asm
; Build with: nasm -f elf64 backends/fft_asm_avx.asm -o backends/fft_asm_avx.o

global fft_1d_asm_avx
global bit_reverse_asm_avx

section .text

; ----------------------------------------------------------------------
; void bit_reverse_asm_avx(Complex *x, int n)
; rdi = x, rsi = n
; ----------------------------------------------------------------------
bit_reverse_asm_avx:
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

        ; temp = x[i]
        vmovsd  xmm0, [rbx + rax]        ; temp_r
        vmovsd  xmm1, [rbx + rax + 8]    ; temp_i

        ; x[i] = x[j]
        vmovsd  xmm2, [rbx + rdx]        ; x[j]_r
        vmovsd  xmm3, [rbx + rdx + 8]    ; x[j]_i
        vmovsd  [rbx + rax],     xmm2
        vmovsd  [rbx + rax + 8], xmm3

        ; x[j] = temp
        vmovsd  [rbx + rdx],     xmm0
        vmovsd  [rbx + rdx + 8], xmm1

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
    vzeroupper
    ret


; ----------------------------------------------------------------------
; void fft_1d_asm_avx(Complex *x, int n, int inverse)
; rdi = *x, rsi = n, rdx = inverse
; ----------------------------------------------------------------------
fft_1d_asm_avx:
    ;--------------- PROLOGO ---------------
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8                    ; 8-byte scratch for x87 <-> SSE moves
    ;--------------- PROLOGO ---------------

    mov rbx, rdi                  ; rbx = *x
    mov r12, rsi                  ; r12 = n
    mov r13, rdx                  ; r13 = inverse

    ; In-place bit reversal
    call bit_reverse_asm_avx

    mov     r14, 2                ; len = 2
out_loop:
        cmp     r14, r12          ; while (len <= n)
        jg      inverse

        ; ------------------- twiddle base: w = cos(angle) + i sin(angle) -------------------
        ; angle = 2π/len * (inverse ? +1 : -1)
        fldpi                           ; ST0 = π
        fadd    st0, st0                ; ST0 = 2π
        mov     [rsp], r14              ; save len (int64)
        fild    qword [rsp]             ; ST0 = (double)len, ST1 = 2π
        fdivp   st1, st0                ; ST0 = 2π/len

        test    r13, r13
        jnz     .declarar_w
        fchs                           ; if (!inverse) angle = -angle

.declarar_w:
        fld     st0                     ; duplicate angle
        fsin                            ; ST0 = sin(angle), ST1 still angle
        fstp    qword [rsp]             ; spill sin
        vmovsd  xmm7, [rsp]             ; w_i in xmm7.low
        fcos                            ; ST0 = cos(angle)
        fstp    qword [rsp]             ; spill cos
        vmovsd  xmm6, [rsp]             ; w_r in xmm6.low
        ; pack w = {w_r, w_i} in xmm6
        unpcklpd xmm6, xmm7             ; xmm6 = [w_r, w_i]
        ; ------------------- end twiddle base -------------------

        ; -------- per-stage constants --------
        mov     r9, r14                 ; r9 = len
        shr     r9, 1                   ; r9 = len/2 (j max)
        mov     r11, r9
        shl     r11, 4                  ; r11 = (len/2) * 16 bytes
        ; -------------------------------------

        ; ---- compute w2 = w*w (complex square), broadcast to both lanes in ymm15 ----
        vpermilpd xmm4,  xmm6, 0x0      ; [w_r, w_r]
        vpermilpd xmm5,  xmm6, 0x3      ; [w_i, w_i]
        vpermilpd xmm12, xmm6, 0x1      ; [w_i, w_r] (swap)
        movapd   xmm10, xmm6            ; copy w
        mulpd    xmm10, xmm4            ; a = w * [w_r,w_r] = [wr*wr, wi*wr]
        mulpd    xmm12, xmm5            ; b = [wi*wi, wr*wi]
        addsubpd xmm10, xmm12           ; w2 = [wr^2 - wi^2, 2*wr*wi]
        vxorpd   ymm15, ymm15, ymm15
        vinsertf128 ymm15, ymm15, xmm10, 0
        vinsertf128 ymm15, ymm15, xmm10, 1
        ; ------------------------------------------------------------------------------

        xor     r15, r15                ; i = 0
mid_loop:
        ; ------- initialize WN as two-lane vector: lane0=1+0i, lane1=w -------
        vxorpd   xmm9,  xmm9,  xmm9         ; 0.0
        fld1
        fstp     qword [rsp]
        vmovsd   xmm8,  [rsp]               ; xmm8.low = 1.0
        unpcklpd xmm8,  xmm9                ; xmm8 = [1.0, 0.0]
        vxorpd   ymm8,  ymm8,  ymm8
        vinsertf128 ymm8, ymm8, xmm8, 0     ; lane0 = 1+0i
        vinsertf128 ymm8, ymm8, xmm6, 1     ; lane1 = w
        ; ----------------------------------------------------------------------

        ; base del bloque i
        mov     rax, r15
        shl     rax, 4
        lea     r10, [rbx + rax]            ; r10 = &x[i]

        ; Pair loop: handle 2 butterflies per iter.
        mov     r8, r9
        and     r8, -2                      ; j_end_even = (len/2) & ~1
        xor     rcx, rcx                    ; j = 0

in_loop:
        cmp     rcx, r8
        jge     .after_pairs

        ; rdi = &x[i+j], rsi = &x[i+j+len/2]
        mov     rdx, rcx
        shl     rdx, 4                      ; j * 16
        lea     rdi, [r10 + rdx]
        lea     rsi, [rdi + r11]

        ; Load two complexes at once (u0,u1) and (t0,t1)
        vmovupd ymm0, [rdi]                 ; ymm0 = [u0_r,u0_i,u1_r,u1_i]
        vmovupd ymm2, [rsi]                 ; ymm2 = [t0_r,t0_i,t1_r,t1_i]

        ; ---------- v = t * WN (complex, lane-wise) ----------
        vpermilpd ymm4,  ymm8,  0x0         ; wr per lane
        vpermilpd ymm5,  ymm8,  0x3         ; wi per lane
        vmulpd   ymm11, ymm2,  ymm4         ; a = t * wr
        vpermilpd ymm12, ymm2,  0x1         ; swap(t): [t_i,t_r | t_i,t_r]
        vmulpd   ymm12, ymm12, ymm5         ; b = swap(t) * wi
        vaddsubpd ymm11, ymm11, ymm12       ; v = [real, imag | real, imag]
        ; ------------------------------------------------------

        ; x[i+j] = u + v   ;   x[i+j+len/2] = u - v
        vmovapd  ymm13, ymm0
        vaddpd   ymm13, ymm13, ymm11
        vmovupd  [rdi], ymm13
        vsubpd   ymm0,  ymm0,  ymm11
        vmovupd  [rsi], ymm0

        ; --------- advance WN by two: WN *= w2 (lane-wise) ---------
        vpermilpd ymm4,  ymm15, 0x0         ; w2r
        vpermilpd ymm5,  ymm15, 0x3         ; w2i
        vmulpd   ymm13, ymm8,  ymm4         ; a = WN * w2r
        vpermilpd ymm14, ymm8,  0x1         ; swap(WN)
        vmulpd   ymm14, ymm14, ymm5         ; b = swap(WN) * w2i
        vaddsubpd ymm8,  ymm13, ymm14       ; WN = a (+/-) b
        ; -----------------------------------------------------------

        add     rcx, 2
        jmp     in_loop

.after_pairs:
        ; Tail if (len/2) is odd: one last butterfly (XMM)
        cmp     rcx, r9
        jge     .end_inner

        mov     rdx, rcx
        shl     rdx, 4
        lea     rdi, [r10 + rdx]
        lea     rsi, [rdi + r11]

        vmovupd xmm0, [rdi]                 ; u
        vmovupd xmm2, [rsi]                 ; t
        vextractf128 xmm8, ymm8, 0          ; wn_j (lane0)

        vpermilpd xmm4,  xmm8,  0x0         ; wr
        vpermilpd xmm5,  xmm8,  0x3         ; wi
        vmulpd    xmm11, xmm2,  xmm4        ; a = t*wr
        vpermilpd xmm12, xmm2,  0x1         ; swap(t)
        vmulpd    xmm12, xmm12, xmm5        ; b = swap(t)*wi
        vaddsubpd xmm11, xmm11, xmm12       ; v

        vmovapd  xmm13, xmm0
        vaddpd   xmm13, xmm13, xmm11        ; u+v
        vmovupd  [rdi], xmm13
        vsubpd   xmm0,  xmm0,  xmm11        ; u-v
        vmovupd  [rsi], xmm0

.end_inner:
        ; i += len
        add     r15, r14
        cmp     r15, r12
        jl      mid_loop

        ; len <<= 1 and next stage
        shl     r14, 1
        jmp     out_loop

; --------------------------- inverse scaling ---------------------------
inverse:
        ; If not inverse, skip 1/n scaling
        test    r13, r13
        jz      end_fft

        ; broadcast (double)n into ymm10
        vcvtsi2sd xmm10, xmm10, r12
        vbroadcastsd ymm10, xmm10          ; ymm10 = [n,n,n,n]

        ; i = 0; process two complexes per iter (32 bytes)
        xor     r15, r15
.inv_loop_pairs:
        ; if (i+1 >= n) goto tail
        mov     rax, r15
        add     rax, 1
        cmp     rax, r12
        jge     .inv_tail

        ; addr = &x[i]
        mov     rax, r15
        shl     rax, 4                      ; i*16
        vmovupd ymm0, [rbx + rax]           ; two complexes
        vdivpd  ymm0, ymm0, ymm10
        vmovupd [rbx + rax], ymm0
        add     r15, 2
        jmp     .inv_loop_pairs

.inv_tail:
        ; If i < n, scale the last complex (scalar XMM path)
        cmp     r15, r12
        jge     end_fft
        mov     rax, r15
        shl     rax, 4
        vmovupd xmm0, [rbx + rax]
        vdivpd  xmm0, xmm0, xmm10
        vmovupd [rbx + rax], xmm0

end_fft:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    vzeroupper
    ret