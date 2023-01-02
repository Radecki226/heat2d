#pragma once 

/* Iterations */
#define MAX_ITER 3000
#define INITIAL 0

/* Dimenstions */
#define WIDTH 256
#define HEIGHT 208
#define N_POINTS (WIDTH*HEIGHT)

/* Coefficients */
#define DX 1
#define DY 1
#define DT (0.1)
#define ALPHA 2
#define GAMMA (ALPHA * (DT/(DX*DX)))

/* Boundaries */
#define TOP 100
#define BOTTOM 100
#define LEFT 100
#define RIGHT 100
