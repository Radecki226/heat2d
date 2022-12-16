#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "my_timers.h"

/*Iterations*/
#define MAX_ITER 1000
#define INITIAL 0

/*Dimenstions*/
#define WIDTH 8002
#define HEIGHT 2002
#define N_POINTS (WIDTH*HEIGHT)

/*Coefficients*/
#define DX 1
#define DY DX
#define DT (0.1)
#define ALPHA 2
#define GAMMA (ALPHA * (DT/(DX*DX)))

/*Boundaries*/
#define TOP 100
#define BOTTOM 0
#define LEFT 0
#define RIGHT 0

/*Parallel part*/
#define ROOT 0


double calc_single_element(double* u, int y, int x){
    return GAMMA * (u[(y+1)*WIDTH + x] + u[(y-1)*WIDTH + x] + u[y*WIDTH + x + 1] 
           + u[y*WIDTH + x - 1] - 4 * u[y*WIDTH + x]) + u[y*WIDTH + x];
}

/*
 * out - output grid -> size (last-first) * Height
 * in - inpput full grid
 * params - information about output chunk of the grid
*/
void calc_one_iter(double *out, double *in, int first_row, int n_rows)
{
    /*Calculate Given Chunk*/
    for (int y = 0; y < n_rows; y++) {
        for (int x = 1; x < WIDTH-1; x++) { /*Columns 0 and ROW_WIDTH-1 are constant*/
            out[y*WIDTH + x] = calc_single_element(in, y + first_row, x);
        }

        /*Boundaries*/
        out[y*WIDTH] = in[(y + first_row) * WIDTH];
        out[(y+1)*WIDTH-1] = in[(y + first_row + 1) * WIDTH - 1];
    }
}

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*Full Grid*/
    double u[N_POINTS];

    /*Full gird wihout top and bottom*/
    double u_msg[(HEIGHT-2) * WIDTH];

    /*Calculate on which positions each thread will operate*/
    int n_rows = (HEIGHT-2) / size;
    int first_row = rank * n_rows + 1;
    int n_points_one_thread = (HEIGHT-2) * WIDTH / size;

    /*Chunk on which particular thread will operate*/
    double *chunk = (double*)malloc(n_points_one_thread * sizeof(double));

    /*Init*/
    if (rank == ROOT) {
        for (int i = 0; i <  N_POINTS; i++) {
            u[i] = INITIAL;
        }

        for (int x = 0; x < WIDTH; x++) {
            u[x] = TOP;
            u[WIDTH*(HEIGHT-1) + x] = BOTTOM;
        }

        for (int y = 1; y < HEIGHT-1; y++) {
            u[y*WIDTH] = LEFT;
            u[y*WIDTH + WIDTH-1] = RIGHT;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == ROOT) {
        start_time();
    }

    /*Calculate Grid MAX_ITER times*/
    for (int t = 0; t < MAX_ITER; t++) {

        /*Broadcast*/
        MPI_Bcast(u, N_POINTS, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

        /*Compute Chunk*/
        calc_one_iter(chunk, u, first_row, n_rows);


        /*Gather all data in root*/
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(chunk, n_points_one_thread, MPI_DOUBLE, u_msg, n_points_one_thread, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);


        /*Copy u_msg to u*/
        if (rank == ROOT) {
            memcpy(u + WIDTH, u_msg, sizeof(u_msg));
            printf("iter = %d\n", t);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /*if (rank == ROOT) {
        printf("[");
        for (int y = 0; y < HEIGHT; y++) {
            printf("[");
            for (int x = 0; x < WIDTH; x++) {
                printf("%f, ", u[y*WIDTH + x]);
            }
            printf("],\n");
        }
        printf("]\n");
    }*/

    if (rank == ROOT) {
        stop_time();
        print_time("Elapsed:");
    }

    MPI_Finalize();

}
