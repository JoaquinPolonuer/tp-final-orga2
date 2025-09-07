global fft_1d_asm
global bit_reverse_asm

; Macro para multiplicación compleja: result = a * b
; Parámetros: a, b, result
; Fórmula: (a_r + a_i*i) * (b_r + b_i*i) = (a_r*b_r - a_i*b_i) + (a_r*b_i + a_i*b_r)*i
%macro COMPLEX_MUL 3
    movupd  %3, %1                      ; t1 = a
    mulpd   %3, %2                      ; t1 = [ar*br, ai*bi]
    xorpd   %3, [rel COMPLEX_NEGHI]     ; t1 = [ar*br, -(ai*bi)]

    movupd  xmm15, %1
    shufpd  xmm15, xmm15, 1   ; xmm15 = [ai, ar]
    mulpd   xmm15, %2         ; xmm15 = [ai*br, ar*bi]

    haddpd  %3, xmm15         ; %3 = [ar*br - ai*bi, ai*br + ar*bi]
%endmacro

; void bit_reverse_asm(Complex *x, int n)
; rdi = x, rsi = n
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

        ; temp = x[i]
        movupd  xmm0, [rbx + rax]        ; temp (real + imag)

        ; x[i] = x[j]
        movupd  xmm2, [rbx + rdx]        ; x[j] (real + imag)
        movupd  [rbx + rax], xmm2

        ; x[j] = temp
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

; void fft_1d_asm(Complex *x, int n, int inverse)
; rdi = *x, rsi = n, rdx = inverse
fft_1d_asm:
    ;--------------- PROLOGO ---------------
    push rbp
	mov rbp, rsp
	push rbx
	push r12
	push r13
	push r14
	push r15
	sub rsp, 8
    ;--------------- PROLOGO ---------------

    mov rbx, rdi 	; rbx = *x
	mov r12, rsi 	; r12 = n
	mov r13, rdx 	; r13 = inverse

    ; Ya estan bien puestos los parametros rdi = *x y rsi = n
    call bit_reverse_asm    ; como hace todo in-place, no hay que hacer nada

    mov     r14, 2                         ; len = 2
    .out_loop:
        cmp     r14, r12                       ; Chequeo si len > n
        jg      .inverse                       ; si la respuesta es si, termina el ciclo

        ; angle = 2π/len * (inverse ? +1 : -1)
        ; Calculamos w = cos(angle) + i sin(angle) con x87 para evitar tablas/constantes en memoria.
        
        ; st0 = 2pi
        fldpi                                   ; st0 = π
        fadd    st0, st0                        ; st0 = 2π

        mov     [rsp], r14                      ; guardar len (int64) en el scratch de 8 bytes
        fild    qword [rsp]                     ; st0 = (double)len, st1 = 2π
        fdivp   st1, st0                        ; st0 = 2π/len

        test    r13, r13
        jnz     .declarar_w                     ; Si inverse es 1, seguimos
        fchs                                    ; Si inverse es 0, fchs (float change sign) cambia el signo de st0
        
        ; Esta seccion es el equivalente a Complex w = {cos(angle), sin(angle)};
        .declarar_w:
        fld     st0                             ; Copio el angulo devuelta en st0, st1 = angulo
        fsin                                    ; st0 = sin(ang)   (ángulo sigue en st1)
        fstp    qword [rsp]                     ; guardar sin en memoria
        movhpd  xmm6, [rsp]                     ; xmm6 = [?, w_i]

        fcos                                    ; st0 = cos(ang)
        fstp    qword [rsp]                     ; guardar cos
        movlpd   xmm6, [rsp]                     ; xmm6 = [w_r, w_i]        ; (pila x87 vacía)

        ; -------- Calculitos que voy a usar en el in_loop ---------
        mov     r9, r14                         ; r9  = len
        shr     r9, 1                           ; r9  = len/2 (contador j max)
        mov     r11, r9                         ; r11 = len/2
        shl     r11, 4                          ; r11 = (len/2)*16 bytes = (len/2) * sizeof(Complex) 
        ; --------------------- Fin calculitos ---------------------

        xor     r15, r15                        ; i = 0
        .mid_loop:
            ; ------- wn = 1 + 0i -------
            pxor    xmm8, xmm8            ; xmm8 = [0.0, 0.0]
            fld1
            fstp    qword [rsp]
            movsd   xmm8, [rsp]           ; xmm8 = [1.0, 0.0]
            ; ------- Fin wn = 1 + 0i -------

            ; base del bloque i
            mov     rax, r15                        ; rax = i
            shl     rax, 4                          ; rax = i * 16
            lea     r10, [rbx + rax]                ; r10 = &x[i]

            xor     rcx, rcx                        ; j = 0
            .in_loop:
                mov     rdx, rcx                        ; rdx = j
                shl     rdx, 4                          ; rdx = j * 16 (porque estamos operando con punteros a complejos)
                lea     rdi, [r10 + rdx]                ; rdi = &x[i + j]
                lea     rsi, [rdi + r11]                ; rsi = &x[i + j + len/2]

                ; Cargar u = (u_r, u_i), t = x[i + j + len/2] = (t_r, t_i)
                movupd  xmm0, [rdi]                     ; xmm0 = {u_r, u_i}
                movupd   xmm2, [rsi]                     ; xmm2 = {t_r, t_i}
                
                ; ----- Complex v = complex_mul(x[i + j + len / 2], wn) -----
                COMPLEX_MUL xmm2, xmm8, xmm4
                ; -----------------------------------------------------------

                ; --------------- x[i + j] = complex_add(u, v) --------------
                movupd  xmm11, xmm0                     ; xmm11 = u_r, u_i
                addpd   xmm11, xmm4                     ; xmm11 = u_r + v_r, u_i + v_i
                movupd  [rdi],   xmm11
                ; -----------------------------------------------------------

                ; --------- x[i + j + len / 2] = complex_sub(u, v) ----------
                subpd   xmm0, xmm4                      ; xmm0 = u_r - v_r, u_i - v_i
                movupd  [rsi],   xmm0
                ; -----------------------------------------------------------

                ; ------------------ wn = complex_mul(wn, w) ----------------
                COMPLEX_MUL xmm8, xmm6, xmm11
                movupd  xmm8, xmm11                     ; xmm8 = wn = new_wn
                ; -----------------------------------------------------------

                ; Guarda de in_loop
                inc     rcx             ; j++
                cmp     rcx, r9         ; Comparo j con len/2
                jl      .in_loop     ; si j < len/2 sigue el loop

            ; Guarda de mid_loop
            add     r15, r14        ; i += len
            cmp     r15, r12        ; Comparo i con n
            jl      .mid_loop       ; Si i < n sigue el loop

        ; "Guarda" de out_loop
        shl     r14, 1          ; len <<= 1
        jmp     .out_loop     ; Sigue el loop, la condicion se chequea arriba
    
    .inverse:
        ; Si no es inversa, saltar el escalado 1/n
        test    r13, r13
        jz      .end_fft

        ; scale = (double)n  (usamos xmm10 como divisor)
        cvtsi2sd xmm10, r12
        shufpd   xmm10, xmm10, 0

        ; i = 0
        xor     r15, r15
    .inverse_loop:
        ; if (i >= n) break
        cmp     r15, r12
        jge     .end_fft

        ; addr = &x[i]  (Complex ocupa 16 bytes: real, imag)
        mov     rax, r15
        shl     rax, 4                      ; i * 16

        ; cargar x[i]_r e imag
        movupd   xmm0, [rbx + rax]           ; real

        ; dividir por n
        divpd   xmm0, xmm10

        ; guardar de vuelta
        movupd   [rbx + rax],     xmm0

        ; i++
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

section .data align=16
COMPLEX_NEGHI: dq 0x0000000000000000, 0x8000000000000000