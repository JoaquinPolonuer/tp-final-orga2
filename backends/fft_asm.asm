global fft_1d_asm
global bit_reverse_asm

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
        movsd   xmm0, [rbx + rax]        ; temp.real
        movsd   xmm1, [rbx + rax + 8]    ; temp.imag

        ; x[i] = x[j]
        movsd   xmm2, [rbx + rdx]        ; x[j].real
        movsd   xmm3, [rbx + rdx + 8]    ; x[j].imag
        movsd   [rbx + rax],     xmm2
        movsd   [rbx + rax + 8], xmm3

        ; x[j] = temp
        movsd   [rbx + rdx],     xmm0
        movsd   [rbx + rdx + 8], xmm1

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
    .outer_loop:
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
        jnz     .declarar_w                       ; Si inverse es 1, seguimos
        fchs                                    ; Si inverse es 0, fchs (float change sign) cambia el signo de st0
        
        ; Esta seccion es el equivalente a Complex w = {cos(angle), sin(angle)};
        .declarar_w:
            fld     st0                             ; Copio el angulo devuelta en st0, st1 = angulo
            fsin                                    ; st0 = sin(ang)   (ángulo sigue en st1)
            fstp    qword [rsp]                     ; guardar sin en memoria
            movsd   xmm7, [rsp]                     ; w.imag = sin(ang)
            fcos                                    ; st0 = cos(ang)
            fstp    qword [rsp]                     ; guardar cos
            movsd   xmm6, [rsp]                     ; w.real = cos(ang)
            ; (pila x87 vacía)

        ; Pre-cálculos para el doble bucle
        mov     r9, r14                         ; r9  = len
        shr     r9, 1                           ; r9  = len/2 (contador j max)
        mov     r11, r9
        shl     r11, 4                          ; r11 = (len/2) * sizeof(Complex) = (len/2)*16 bytes

        xor     r15, r15                        ; i = 0
        .mid_loop:
            ; wn = 1 + 0i
            pxor    xmm9, xmm9                      ; wn.imag = 0.0
            fld1
            fstp    qword [rsp]
            movsd   xmm8, [rsp]                     ; wn.real = 1.0

            ; base del bloque i
            mov     rax, r15
            shl     rax, 4                          ; i * 16
            lea     r10, [rbx + rax]                ; r10 = &x[i]

            xor     rcx, rcx                        ; j = 0
            .inner_loop:
                ; Direcciones de u = x[i + j] y t = x[i + j + len/2]
                mov     rdx, rcx
                shl     rdx, 4                          ; j * 16
                lea     rdi, [r10 + rdx]                ; rdi = &x[i + j]
                lea     rsi, [rdi + r11]                ; rsi = &x[i + j + len/2]

                ; Cargar u = (u_r, u_i)
                movsd   xmm0, [rdi]                     ; u_r
                movsd   xmm1, [rdi+8]                   ; u_i

                ; Cargar t = x[i + j + len/2] = (t_r, t_i)
                movsd   xmm2, [rsi]                     ; t_r
                movsd   xmm3, [rsi+8]                   ; t_i

                ; v = t * wn = (t_r*wn_r - t_i*wn_i, t_r*wn_i + t_i*wn_r)
                movapd  xmm4, xmm2
                mulsd   xmm4, xmm8                      ; t_r * wn_r
                movapd  xmm5, xmm3
                mulsd   xmm5, xmm9                      ; t_i * wn_i
                subsd   xmm4, xmm5                      ; v_r

                movapd  xmm5, xmm2
                mulsd   xmm5, xmm9                      ; t_r * wn_i
                movapd  xmm11, xmm3
                mulsd   xmm11, xmm8                     ; t_i * wn_r
                addsd   xmm5, xmm11                     ; v_i

                ; x[i + j]         = u + v
                movapd  xmm11, xmm0
                addsd   xmm11, xmm4                     ; u_r + v_r
                movapd  xmm12, xmm1
                addsd   xmm12, xmm5                     ; u_i + v_i
                movsd   [rdi],   xmm11
                movsd   [rdi+8], xmm12

                ; x[i + j + len/2] = u - v
                subsd   xmm0, xmm4                      ; u_r - v_r
                subsd   xmm1, xmm5                      ; u_i - v_i
                movsd   [rsi],   xmm0
                movsd   [rsi+8], xmm1

                ; wn *= w
                movapd  xmm11, xmm8
                mulsd   xmm11, xmm6                     ; wn_r * w_r
                movapd  xmm12, xmm9
                mulsd   xmm12, xmm7                     ; wn_i * w_i
                subsd   xmm11, xmm12                    ; new_r

                movapd  xmm13, xmm8
                mulsd   xmm13, xmm7                     ; wn_r * w_i
                movapd  xmm14, xmm9
                mulsd   xmm14, xmm6                     ; wn_i * w_r
                addsd   xmm13, xmm14                    ; new_i

                movapd  xmm8, xmm11                     ; wn.real = new_r
                movapd  xmm9, xmm13                     ; wn.imag = new_i

                ; j++
                inc     rcx
                cmp     rcx, r9
                jl      .inner_loop

        ; i += len
        add     r15, r14
        cmp     r15, r12
        jl      .mid_loop

        ; len <<= 1 y siguiente etapa
        shl     r14, 1
        jmp     .outer_loop
    
    .inverse:
        ; Si no es inversa, saltar el escalado 1/n
        test    r13, r13
        jz      .end_fft

        ; scale = (double)n  (usamos xmm10 como divisor)
        cvtsi2sd xmm10, r12

        ; i = 0
        xor     r15, r15
    .inverse_loop:
        ; if (i >= n) break
        cmp     r15, r12
        jge     .end_fft

        ; addr = &x[i]  (Complex ocupa 16 bytes: real, imag)
        mov     rax, r15
        shl     rax, 4                      ; i * 16

        ; cargar x[i].real e imag
        movsd   xmm0, [rbx + rax]           ; real
        movsd   xmm1, [rbx + rax + 8]       ; imag

        ; dividir por n
        divsd   xmm0, xmm10
        divsd   xmm1, xmm10

        ; guardar de vuelta
        movsd   [rbx + rax],     xmm0
        movsd   [rbx + rax + 8], xmm1

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