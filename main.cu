#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    size_t A_sz, B_sz, C_sz;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    dim3 dim_grid, dim_block;

    if (argc == 1) {
        matArow = 1000;
        matAcol = matBrow = 1000;
        matBcol = 1000;
    } else if (argc == 2) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[1]);
        matBcol = atoi(argv[1]);
    } else if (argc == 4) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[2]);
        matBcol = atoi(argv[3]);
    } else {
        printf("\n    Invalid input parameters!"
      "\n    Usage: ./sgemm-tiled                # All matrices are 1000 x 1000"
      "\n    Usage: ./sgemm-tiled <m>            # All matrices are m x m"
      "\n    Usage: ./sgemm-tiled <m> <k> <n>    # A: m x k, B: k x n, C: m x n"
      "\n");
        exit(0);
    }
   
    A_sz = matArow*matAcol;
    B_sz = matBrow*matBcol;
    C_sz = matArow*matBcol;

    A_h = (float*) malloc( sizeof(float)*A_sz );
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc( sizeof(float)*B_sz );
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    C_h = (float*) malloc( sizeof(float)*C_sz );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matArow, matAcol, matBrow, matBcol, matArow, matBcol);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //Allocate device memory
    cudaMalloc((void**) &A_d, sizeof(float)*A_sz);
    cudaMalloc((void**) &B_d, sizeof(float)*B_sz);
    cudaMalloc((void**) &C_d, sizeof(float)*C_sz);
	
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    float copy_to_gpu, kernel_computation, copy_from_gpu;
    // Copy host variables to device ------------------------------------------
    startTime(&timer);
	
    cudaMemcpy(A_d, A_h, sizeof(float)*A_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(float)*B_sz, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer); 
    copy_to_gpu = elapsedTime(timer);

    // Launch kernel using standard sgemm interface ---------------------------
    startTime(&timer);
    basicSgemm(matArow, matBcol, matBrow, A_d, B_d, C_d);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");
    stopTime(&timer); 
    kernel_computation = elapsedTime(timer);

    // Copy device variables from host ----------------------------------------
    startTime(&timer);

    cudaMemcpy(C_h, C_d, sizeof(float)*C_sz, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    stopTime(&timer); 
    copy_from_gpu = elapsedTime(timer);
    printf("Total time for naive kernel: %f, copy to gpu: %f, kernel: %f, copy from gpu: %f\n", copy_to_gpu + kernel_computation + copy_from_gpu, copy_to_gpu, kernel_computation, copy_from_gpu);


    /* LAUNCH OPTIMIZED KERNEL NOW */
    cudaFree(C_d);
    cudaMalloc((void**) &C_d, sizeof(float)*C_sz);
    cudaDeviceSynchronize();

    startTime(&timer);
    tiledSgemm(matArow, matBcol, matBrow, A_d, B_d, C_d);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");
    stopTime(&timer); 
    kernel_computation = elapsedTime(timer);

    // Copy device variables from host ----------------------------------------
    startTime(&timer);

    cudaMemcpy(C_h, C_d, sizeof(float)*C_sz, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    stopTime(&timer); 
    copy_from_gpu = elapsedTime(timer);

    printf("Total time for optimized kernel: %f, copy to gpu: %f, kernel: %f, copy from gpu: %f\n", copy_to_gpu + kernel_computation + copy_from_gpu, copy_to_gpu, kernel_computation, copy_from_gpu);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    /* CUBLAS MATRIX MULTIPLY, THE BIG APPLE OF MATMUL */
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMalloc((void**) &A_d, sizeof(float)*A_sz);
    cudaMalloc((void**) &B_d, sizeof(float)*B_sz);
    cudaMalloc((void**) &C_d, sizeof(float)*C_sz);

    startTime(&timer);
    // Copy matrices to device
    cublasSetMatrix(matArow, matAcol, sizeof(float), A_h, matArow, A_d, matArow);
    cublasSetMatrix(matBrow, matBcol, sizeof(float), B_h, matBrow, B_d, matBrow);

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        matArow, matBcol, matBrow,
        &alpha,
        A_d, matArow,  // Leading dimension of A
        B_d, matBrow,  // Leading dimension of B
        &beta,
        C_d, matArow); // Leading dimension of C
    cudaDeviceSynchronize();
    stopTime(&timer);
    kernel_computation = elapsedTime(timer);
    
    startTime(&timer);
    cublasGetMatrix(matArow, matBcol, sizeof(float), C_d, matArow, C_h, matArow);
    cudaDeviceSynchronize();
    stopTime(&timer);
    copy_from_gpu = elapsedTime(timer);

    printf("Total time for cuBLAS kernel: %f, copy to and kernel: %f, copy from GPU: %f\n", kernel_computation + copy_from_gpu, kernel_computation, copy_from_gpu);

    cublasDestroy(handle);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    // Free memory ------------------------------------------------------------
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}