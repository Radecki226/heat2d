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
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (y > 0 && y < HEIGHT-1)
    {
        if (x > 0 && x < WIDTH-1)
        {
            float uij   = in[get_index(y, x, WIDTH)];
            float uim1j = in[get_index(y-1, x, WIDTH)];
            float uijm1 = in[get_index(y, x-1, WIDTH)];
            float uip1j = in[get_index(y+1, x, WIDTH)];
            float uijp1 = in[get_index(y, x+1, WIDTH)];

            out[get_index(y, x, WIDTH)] = GAMMA*(uip1j+uim1j+uijp1+uijm1-4*uij)+uij;

        }
    }

    /* Take care of boundaries */
    out[get_index(y, 0, WIDTH)]  = LEFT;
    out[get_index(y, -1, WIDTH)] = RIGHT;
    out[get_index(0, x, 0)] = TOP;
    out[get_index(HEIGHT-1, x, WIDTH)] = BOTTOM;

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
    //print_result(u0);

    /* Release the memory */
    free(u0);
    cudaFree(d_u0);
    cudaFree(d_u1);

    return 0;
}
