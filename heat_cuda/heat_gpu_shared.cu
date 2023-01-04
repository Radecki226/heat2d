#include <algorithm>
#include <stdio.h>
#include "heat.h"
#include "heat_utils.h"

/* CUDA parameters */
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

/* Get 1D index of 2D array */
int __host__ __device__ get_index(const int y, const int x, const int width)
{
    return y*width + x;
}

__global__ void heat_kernel(const float* in, float* out)
{
    __shared__ float shared_buffer[(BLOCK_SIZE_X + 2)*(BLOCK_SIZE_Y + 2)];
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    int s_x = threadIdx.x + 1;
    int s_y = threadIdx.y + 1;
    int s_n = BLOCK_SIZE_Y + 2;

    int global_idx = get_index(y, x, WIDTH);
    int shared_idx = get_index(s_y, s_x, s_n);
    shared_buffer[shared_idx] = in[global_idx]; // Central element

    /* 2D Heat Equation */
    if (y > 0 && y < HEIGHT-1)
    {
        if (x > 0 && x < WIDTH-1)
        {
            /* Load data into shared memory */
            if (s_y == 1)               // Top element
            {
                shared_buffer[shared_idx-s_n] = in[global_idx-WIDTH];
            }
            if (s_y == BLOCK_SIZE_X)    // Bottom element
            {
                shared_buffer[shared_idx+s_n] = in[global_idx+WIDTH];
            }
            if (s_x == 1)               // Left element
            {
                shared_buffer[shared_idx-1] = in[global_idx-1];
            }
            if (s_x == BLOCK_SIZE_Y)    // Right element
            {
                shared_buffer[shared_idx+1] = in[global_idx+1];
            }

            /* Make sure all the data is loaded at this point */
            __syncthreads();

            float uij   = shared_buffer[get_index(s_y, s_x, s_n)];
            float uim1j = shared_buffer[get_index(s_y-1, s_x, s_n)];
            float uijm1 = shared_buffer[get_index(s_y, s_x-1, s_n)];
            float uip1j = shared_buffer[get_index(s_y+1, s_x, s_n)];
            float uijp1 = shared_buffer[get_index(s_y, s_x+1, s_n)];

            out[get_index(y, x, WIDTH)] = GAMMA*(uip1j+uim1j+uijp1+uijm1-4*uij)+uij;
        }
    }
}

int main(void)
{
    float* u0 = (float*)calloc(N_POINTS, sizeof(float));

    /* Initialize the data */
    data_init(u0);

    float* d_u0;
    float* d_u1;

    cudaMalloc((void**)&d_u0, N_POINTS*sizeof(float));
    cudaMalloc((void**)&d_u1, N_POINTS*sizeof(float));

    cudaMemcpy(d_u0, u0, N_POINTS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u1, u0, N_POINTS*sizeof(float), cudaMemcpyHostToDevice);

    dim3 numBlocks(WIDTH/BLOCK_SIZE_X + 1, HEIGHT/BLOCK_SIZE_Y + 1);
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    printf("numBlocks.x: %d, numBlocks.y: %d\n", numBlocks.x, numBlocks.y);
    printf("threadsPerBlock.x: %d, threadsPerBlock.y: %d\n", threadsPerBlock.x, threadsPerBlock.y);

    /* Measure execution time - begin */
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    /* Main loop */
    for (int n = 0; n < MAX_ITER; n++)
    {
        heat_kernel<<<numBlocks, threadsPerBlock>>>(d_u0, d_u1);

        /* Swap pointers for the next iteration */
        std::swap(d_u0, d_u1);
    }

    cudaMemcpy(u0, d_u1, N_POINTS*sizeof(float), cudaMemcpyDeviceToHost);

    /* Measure execution time - end */
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("#2D HEAT EQUATION - CUDA execution time: %f ms\n", time);

    /* Save result to the file */
    print_result(u0);

    /* Release the memory */
    free(u0);
    cudaFree(d_u0);
    cudaFree(d_u1);

    return 0;
}
