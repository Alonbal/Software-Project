
#define PY_SSIZE_T_CLEAN 
#include "symnmf.h"
#include <Python.h>

static PyObject* calc_mat(PyObject *self, PyObject *args) {
    char * filename, * action;
    double ** result;
    int n, i, j;
    PyObject * ret, * col_py;

    if(!PyArg_ParseTuple(args, "ss", &filename, &action)) {
        return NULL;
    }
    
    result = read_and_calculate(filename, action, &n);
    if (result == NULL) return NULL;

    ret = PyList_New(n);
    if (ret == NULL) {
        free_mat(result, n);
        return NULL;
    }

    for (i = 0; i < n; i++) {
        col_py = PyList_New(n);
        if (col_py == NULL) {
            free_mat(result, n);
        }

        for (j=0; j < n; j++) {
            PyList_SetItem(col_py, j, Py_BuildValue("d", result[i][j]));
        }

        PyList_SetItem(ret, i, col_py);
    }

    free_mat(result, n);
    return ret;
}


static PyMethodDef symnmfmoduleMethods[] = {
    {"calc_mat",                   
      (PyCFunction) calc_mat, /* the C-function that implements the Python function and returns static PyObject*  */
      METH_VARARGS,           
      PyDoc_STR("calc_mat(file_name, action) reads the data from file_name, and then returns a proccessed matrix according to the action: \n-sym: similarity matrix \n-ddg: degree matrix \n-norm: normed laplacian")},
    
    {NULL, NULL, 0, NULL}    
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule", 
    "Includes the calc_mat function, see docstring.", 
    -1,  
    symnmfmoduleMethods
};

PyMODINIT_FUNC PyInit_symnmfmodule(void)
{
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (!m) {
        return NULL;
    }
    return m;
}

