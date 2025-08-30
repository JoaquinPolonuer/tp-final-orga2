global fft_1d_asm
extern bit_reverse

fft_1d_asm:
	; rdi = *x
	; rsi = n
	; rdx = inverse

    push rbp
	mov rbp, rsp
	push rbx
	push r12
	push r13
	push r14
	push r15
	sub rsp, 8
    
    mov rbx, rdi 	; rbx = *x
	mov r12, rsi 	; r12 = n
	mov r13, rdx 	; r13 = inverse

    ; Ya estan bien puestos los parametros rdi = *x y rsi = n
    call bit_reverse    ; como hace todo in-place, no hay que hacer nada


    .outer_loop:

        .inner_loop:

    
    .inverse:
        .inverse_loop:


    .end_fft:
        add rsp, 8
        pop r15
        pop r14
        pop r13
        pop r12
        pop rbx
        pop rbp
        ret