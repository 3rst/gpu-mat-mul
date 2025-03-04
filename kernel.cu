#include <stdio.h>

#define TILE_SIZE 32

__global__ void basicsgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    //Naive implementation
    if(row<m && col<n)
    {
        float pValue = 0.0f;
        for(int l=0;l<k;++l)
        {
            pValue+=A[row*k+l]*B[l*n+col];
        }
        C[row*n+col]=pValue;
    }
}

__global__ void tiledsgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE

    __shared__ float A_cache[TILE_SIZE*TILE_SIZE];
    __shared__ float B_cache[TILE_SIZE*TILE_SIZE];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // /*************************************************************************/
    float pValue = 0;
    //You have to load memory on each phase
    for(int p=0; p<((k+blockDim.x-1)/blockDim.x); p++)
    {
        int A_col = threadIdx.x + p*blockDim.x;
        int B_row = threadIdx.y + p*blockDim.y;

        //For each phase
        //Copy elements onto shared memory, A thread can 
        // printf("copying element from A[%d]: %lf when k=%d\n", row*k + (threadIdx.x + p*blockDim.x), A[row*k + (threadIdx.x + p*blockDim.x)], k);
        A_cache[threadIdx.y*blockDim.x + threadIdx.x] = (row<m && A_col<k) ? A[row*k + (threadIdx.x + p*blockDim.x)] : 0;
        B_cache[threadIdx.y*blockDim.x + threadIdx.x] = (col<n && B_row<k) ? B[(threadIdx.y + p*blockDim.y)*n + col] : 0;   

        __syncthreads();
        //Matrix multiply for A_cache and B_cache
        for(int l=0;l<blockDim.x;l++)
        {
            // printf("Multiplying values: A %lf and B %lf\n",A_cache[threadIdx.y*blockDim.x + l],B_cache[l*blockDim.x + threadIdx.x]);
            pValue += A_cache[threadIdx.y*blockDim.x + l] * B_cache[l*blockDim.x + threadIdx.x];
        }
        // printf("In phase: %d, Final value for C[%d]=%lf\n", p, row*n+col, pValue);
        __syncthreads();

        if (row < m && col < n) {
           C[row * n + col] = pValue;
        }
    }
}

//m = Arow, n=Bcol, k=Brow
void tiledSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------
    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    /*************************************************************************/
    //INSERT CODE HERE
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 num_blocks(((n+BLOCK_SIZE-1)/BLOCK_SIZE), ((m+BLOCK_SIZE-1)/BLOCK_SIZE), 1);
    /*************************************************************************/

    // Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
	tiledsgemm <<<num_blocks, threads_per_block>>> (m,n,k,A,B,C);
    /*************************************************************************/
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
        const unsigned int BLOCK_SIZE = TILE_SIZE;
        /*************************************************************************/
        //INSERT CODE HERE
        dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 num_blocks(((n+BLOCK_SIZE-1)/BLOCK_SIZE), ((m+BLOCK_SIZE-1)/BLOCK_SIZE), 1);
        /*************************************************************************/
    
        // Invoke CUDA kernel -----------------------------------------------------
    
        /*************************************************************************/
        //INSERT CODE HERE
        basicsgemm <<<num_blocks, threads_per_block>>> (m,n,k,A,B,C);
        /*************************************************************************/
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
}

