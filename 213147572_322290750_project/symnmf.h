
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

double ** read_from_file(FILE * fp, int * np, int * dp);

double ** similarity_matrix(double ** datapoints, int n, int d);

double ** alloc_matrix(int rowc, int colc);

void free_mat(double ** mat, int n);

double distance_sq(double * p1, double * p2, int d);

double ** diagonal_matrix(double ** sim_mat, int n);

double ** laplacian(double ** diag_mat, double ** sim_mat, int n);

double ** read_and_calculate(char * filename, char * action, int * np);

void print_result(double ** result, int n);
