
#include "symnmf.h"

const char * ERR_MSG = "An Error Has Occurred\n";

int main(int argc, char ** argv) {

    char *goal;
    int n;
    double ** result;

    if (argc != 3) {
        printf("%s", ERR_MSG);
        return 1;
    }

    goal = argv[1];

    if (strcmp(goal, "sym") * strcmp(goal, "norm") * strcmp(goal, "ddg") != 0) {
        printf("%s", ERR_MSG);
        return 1;
    }

    result = read_and_calculate(argv[2], goal, &n);
    if (result == NULL) {
        printf("%s", ERR_MSG);
        return 1;
    }

    print_result(result, n);

    free_mat(result, n);

    return 0;

}

double ** alloc_matrix(int rowc, int colc) {
    int i, j;
    double ** mat = malloc(sizeof(double*) * rowc);

    if (mat == NULL) {
        return NULL;
    }

    for (i = 0; i < rowc; i++) {
        mat[i] = calloc(sizeof(double), colc);
        if (mat[i] == NULL) {
            for (j = 0; j < i; j++)
                free(mat[j]);
            free(mat);
            return NULL;
        }
    }

    return mat;
}

void free_mat(double ** mat, int n) {
    int i = 0;
    for (i = 0; i < n; i++)
        free(mat[i]);
    free(mat);
}

double ** read_from_file(FILE * fp, int * np, int * dp) {
    
    char c;
    double ** datapoints;
    int i, j;

    dp[0] = 1;
    np[0] = 0;

    while ((c = getc(fp)) != '\n') {
        if (c == ',')
            dp[0]++;
    }

    np[0]++;
    while ((c = getc(fp)) != EOF) {
        if (c == '\n')
            np[0]++;
    }

    rewind(fp);

    datapoints = alloc_matrix(np[0], dp[0]);
    if (datapoints == NULL) return NULL;

    for (i = 0; i < np[0]; i++) {
        for (j = 0; j < dp[0]; j++) {
            fscanf(fp, "%lf", &datapoints[i][j]);
            getc(fp);
        }
    }

    return datapoints;

}

double ** similarity_matrix(double ** datapoints, int n, int d) {
    int i, j;
    double value;
    double ** sim_matrix = alloc_matrix(n, n);
    if (sim_matrix == NULL) return NULL;

    for (i = 0; i < n; i++) {
        for (j = 0; j < i; j++) {
            value = exp(-distance_sq(datapoints[i], datapoints[j], d) / 2.0 );
            sim_matrix[i][j] = value;
            sim_matrix[j][i] = value;
        }
        sim_matrix[i][i] = 0;
    }

    return sim_matrix;
}

double ** diagonal_matrix(double ** sim_mat, int n) {
    int i, j;
    double ** diag_mat = alloc_matrix(n, n), sum;
    if (diag_mat == NULL) return NULL;

    for (i = 0; i < n; i++) {
        sum = 0;
        for (j = 0; j < n; j++)
            sum += sim_mat[i][j];
        diag_mat[i][i] = sum;
    }

    return diag_mat;
}

double ** laplacian(double ** diag_mat, double ** sim_mat, int n) {
    int i, j;
    double ** laplace_mat = alloc_matrix(n, n);

    if (laplace_mat == NULL) return NULL;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            laplace_mat[i][j] = sim_mat[i][j] / sqrt(diag_mat[i][i] * diag_mat[j][j]);
        }
    }

    return laplace_mat;

}

double distance_sq(double * p1, double * p2, int d) {
    int i = 0;
    double sum = 0;
    for (; i < d; i++) {
        sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return sum;
}

double ** read_and_calculate(char * filename, char * goal, int * np) {
    
    int n, d;
    FILE * fp;
    double **datapoints, ** sym, ** diag, ** laplace;


    fp = fopen(filename, "r");
    if (fp == NULL) {
        return NULL;
    }

    datapoints = read_from_file(fp, &n, &d);
    
    fclose(fp);

    np[0] = n;

    sym = similarity_matrix(datapoints, n, d);
    free_mat(datapoints, n);

    if (sym == NULL) return NULL;

    if (strcmp(goal, "sym") == 0) {
        return sym;
    }

    diag = diagonal_matrix(sym, n);
    if (diag == NULL) {
        free_mat(sym, n);
        return NULL;
    }

    if (strcmp(goal, "ddg") == 0) {
        free_mat(sym, n);
        return diag;
    }

    laplace = laplacian(diag, sym, n);
    free_mat(sym, n);
    free_mat(diag, n);
    if (laplace == NULL) return NULL;

    if (strcmp(goal, "norm") == 0) {
        return laplace;
    }

    free_mat(laplace, n);

    return NULL;
}

void print_result(double ** result, int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j=0; j < n; j++) {
            printf("%.4f", result[i][j]);
            if (j != n - 1)
                printf(",");
        }
        printf("\n");
    }
}
