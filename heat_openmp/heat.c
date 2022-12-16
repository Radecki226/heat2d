#include <omp.h>
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
 * out - output grid
 * in - input grid
*/
void calc_one_iter(double *out, double *in)
{
    /*Full Grid*/
    for (int y = 1; y < HEIGHT-1; y++) {

        #pragma omp parallel for
        for (int x = 1; x < WIDTH-1; x++){
            out[y*WIDTH + x] = calc_single_element(in, y, x);
        }

        /*Boundaries*/
        out[y*WIDTH] = in[y * WIDTH];
        out[(y+1)*WIDTH-1] = in[(y + 1) * WIDTH - 1];
    }

    /*Boundaries*/
    for (int x = 0; x < WIDTH; x++) {
        out[x] = TOP;
        out[WIDTH*(HEIGHT-1) + x] = BOTTOM;
    }

}

int main(int argc, char **argv)
{
    omp_set_num_threads(1);


    /*Full Grid*/
    double u0[N_POINTS];
    double u1[N_POINTS];

    int mem_sel = 0;

    /*Init*/
    for (int i = 0; i <  N_POINTS; i++) {
        u0[i] = INITIAL;
    }
    for (int x = 0; x < WIDTH; x++) {
        u0[x] = TOP;
        u0[WIDTH*(HEIGHT-1) + x] = BOTTOM;
    }
    for (int y = 1; y < HEIGHT-1; y++) {
        u0[y*WIDTH] = LEFT;
        u0[y*WIDTH + WIDTH-1] = RIGHT;
    }

    start_time();
    /*Calculate Grid MAX_ITER times*/
    for (int t = 0; t < MAX_ITER; t++) {
        
        //Compute Iter
        if (mem_sel == 0) {
            calc_one_iter(u1, u0);
            mem_sel = 1;
        } else {
            calc_one_iter(u0, u1);
            mem_sel = 0;
        }
        
    }
    stop_time();

    double *result;

    if (mem_sel == 1) {
        result = u1;
    } else {
        result = u0;
    }

    /*printf("[");
    for (int y = 0; y < HEIGHT; y++) {
        printf("[");
        for (int x = 0; x < WIDTH; x++) {
            printf("%f, ", result[y*WIDTH + x]);
        }
        printf("],\n");
    }
    printf("]\n");*/

    print_time("Elapsed:");

}
