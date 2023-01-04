#pragma once

#include <stdio.h>

void data_init(float* u0)
{
    for (size_t i = 0; i <  N_POINTS; i++)
    {
        u0[i] = INITIAL;
    }
    for (size_t x = 0; x < WIDTH; x++)
    {
        u0[x] = TOP;
        u0[WIDTH*(HEIGHT-1)+x] = BOTTOM;
    }
    for (size_t y = 1; y < HEIGHT-1; y++)
    {
        u0[y*WIDTH] = LEFT;
        u0[y*WIDTH+WIDTH-1] = RIGHT;
    }
}

void print_result(const float* result)
{
    printf("u = [");
    for (size_t y = 0; y < HEIGHT; y++) {
        printf("[");
        for (size_t x = 0; x < WIDTH; x++) {
            printf("%f, ", result[y*WIDTH + x]);
        }
        printf("],\n");
    }
    printf("]\n");
}
