#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void printpoint(double *point, int D) {
    int i;

    for (i = 0; i < D; i++) {
        printf("%.4f", point[i]);
        if (i < D - 1) printf(",");
    }

}

void printdata(double **data, int N, int D) {
    for (int i = 0; i < N; i++) {
        printpoint(data[i], D);
        printf("\n");
    }
}

void free_data(double **data, int N);

void memcopy(void* target, const void* src, size_t size) {
    char *csrc = (char *)src;
    char *ctarget = (char *)target;
    size_t i;

    for (i = 0; i < size; i++) {
        ctarget[i] = csrc[i];
    }
}

double distance(double *p1, double *p2, int D);

int index_of_closest(double *point, double **clusters_centers, int K, int D);

int init_zeros(double ***points, int K, int D);

void add_to_point(double *target, double *src, int D);

void divide_by(double *target, double num, int D);

void free_data(double **data, int N);

int read_2darray(PyObject* array_py, double *** p_array, int *K, int *D);

int fit_c(double *** cluster_centers, int K, int D,
        double ** data, int N, int iter, double EPS) {

    int error_occurred, i, j, closest_index;
    int  *bucket_sizes;
    double **new_cluster_centers, max_diff = EPS + 1, diff;

    bucket_sizes = malloc(sizeof(int) * K);
    if (bucket_sizes == NULL) {
        return 1;
    }

    error_occurred = init_zeros(&new_cluster_centers, K, D);
    if (error_occurred) {
        return 1;
    }

    while (iter > 0 && max_diff > EPS)
    {
        max_diff = 0;

        for (i = 0; i < K; i++) {
            bucket_sizes[i] = 0;
            for (j = 0; j < D; j++) {
                new_cluster_centers[i][j] = 0;
            }
        }

        for (i = 0; i < N; i++) {
            closest_index = index_of_closest(data[i], *cluster_centers, K, D);
            bucket_sizes[closest_index]++;
            add_to_point(new_cluster_centers[closest_index], data[i], D);
        }


        for (i = 0; i < K; i++) {
            if (bucket_sizes[i] == 0) continue;
            divide_by(new_cluster_centers[i], bucket_sizes[i], D);
            diff = distance(new_cluster_centers[i], (*cluster_centers)[i], D);
            if (diff > max_diff) max_diff = diff;

            memcopy((*cluster_centers)[i], new_cluster_centers[i], D * sizeof(double));
        }

        iter--;
    }

    free_data(new_cluster_centers, K);

    return 0;
}


static PyObject* fit(PyObject *self, PyObject *args) {
    double ** cluster_centers, ** data, EPS;
    PyObject *cluster_centers_py, *data_py, *point_py, *ret;
    int K, D, N, iter, i, j;

    if (!PyArg_ParseTuple(args, "OOid", 
                          &cluster_centers_py, &data_py, &iter, &EPS)) {
        return NULL;
    }

    if (read_2darray(cluster_centers_py, &cluster_centers, &K, &D)) {
        return NULL;
    }

    if (read_2darray(data_py, &data, &N, &D)) {
        free_data(cluster_centers, K);
        free(cluster_centers);
        return NULL;
    }

    if(fit_c(&cluster_centers, K, D, data, N, iter, EPS)) {
        free_data(cluster_centers, K);
        free(cluster_centers);
        return NULL;
    }

    free_data(data, N);

    ret = PyList_New(K);
    if (ret == NULL) {
        free_data(cluster_centers, K);
        return NULL;
    }

    for (i = 0; i < K; i++) {
        point_py = PyList_New(D);
        if (point_py == NULL) {
            free_data(cluster_centers, K);
        }
        for (j=0; j < D; j++) {
            PyList_SetItem(point_py, j, Py_BuildValue("d", cluster_centers[i][j]));
        }

        PyList_SetItem(ret, i, point_py);
    }

    free_data(cluster_centers, K);

    return ret;
}


static PyMethodDef geoMethods[] = {
    {"fit",                   
      (PyCFunction) fit, /* the C-function that implements the Python function and returns static PyObject*  */
      METH_VARARGS,           
      PyDoc_STR("Fit(Init_Clusters, Datapoints, max_iter, Epsilon) runs the kmeans++ on Datapoints with initial centers Init_Clusters, until change less than EPS, or for max_iter iterations")},
    
    {NULL, NULL, 0, NULL}    
};

static struct PyModuleDef mykmeanssp = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp", 
    "Includes the fit functions, see its docstring.", /* module documentation, may be NULL */
    -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    geoMethods /* the PyMethodDef array from before containing the methods of the extension */
};

PyMODINIT_FUNC PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&mykmeanssp);
    if (!m) {
        return NULL;
    }
    return m;
}


int read_2darray(PyObject* array_py, double *** p_array, int *K, int *D) {
    int i, j;
    PyObject *point_py;
    double num, *point;

    K[0] = PyObject_Length(array_py);
    if (K[0] < 0) {
        return 1;
    }
    
    (*p_array) = malloc(sizeof(double*) * K[0]);
    if ((*p_array) == NULL) {
        return 1;
    }

    for (i = 0; i < K[0]; i++) {
        point_py = PyList_GetItem(array_py, i);
        D[0] = PyObject_Length(point_py);
        if (D[0] < 0 || (point = malloc(sizeof(double) * D[0])) == NULL) {
            free_data(*p_array, i);
            free(*p_array);
            return 1;
        }

        for (j = 0; j < D[0]; j++) {
            num = PyFloat_AsDouble(PyList_GetItem(point_py, j));
            point[j] = num;
        }

        (*p_array)[i] = point;
    }

    return 0;
}

void free_data(double **data, int N) {
    int i;
    for (i = 0; i < N; i++) {
        free(data[i]);
    }
    free(data);
}

double distance(double *p1, double *p2, int D) {
    int i;
    double total = 0;

    for (i = 0; i < D; i++) {
        total += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }

    return sqrt(total);
}

int index_of_closest(double *point, double **clusters_centers, int K, int D) {
    int best_ind = 0, curr_ind;
    double best_dist = distance(point, clusters_centers[0], D), curr_dist;

    for (curr_ind = 1; curr_ind < K; curr_ind++) {
        curr_dist = distance(point, clusters_centers[curr_ind], D);
        if (curr_dist < best_dist) {
            best_dist = curr_dist;
            best_ind = curr_ind;
        }
    }

    return best_ind;

}

int init_zeros(double ***points, int K, int D) {
    int i;

    *points = malloc(sizeof(double*) * K);
    if (*points == NULL) {
        return 1;
    }

    for (i = 0; i < K; i++) {
        (*points)[i] = calloc(D, sizeof(double));
        if ((*points)[i] == NULL) {
            free_data((*points), i);
            return 1;
        }
    }

    return 0;
}

void add_to_point(double *target, double *src, int D) {
    int i;

    for (i = 0; i < D; i++) {
        target[i] += src[i];
    }
}

void divide_by(double *target, double num, int D) {
    int i;

    if (num == 0) return;

    for (i = 0; i < D; i++) {
        target[i] /= num;
    }
}


