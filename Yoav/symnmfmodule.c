#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "symnmf.h"

static void free_cords_matrix(struct cord **matrix, int n) {
    int i;
    for (i = 0; i < n; i++) {
        free_cords(matrix[i]);
    }
    free(matrix);
}

static struct vector *linkedList_to_vectors(PyObject *py_vectors) {
    int i, n = PyObject_Length(py_vectors);
    struct vector *head = NULL, *curr = NULL, *new_vec = NULL;
    PyObject* py_vec = NULL;

    if (n <= 0) {
        return NULL;
    }

    for (i = 0; i < n; i++) {
        py_vec = PyList_GetItem(py_vectors, i);
        if(py_vec == NULL) {
            free_vectors(head);
            return NULL;
        }
        new_vec = malloc(sizeof(struct vector));
        if (new_vec  == NULL) {
            free_vectors(head);
            return NULL;
        }
        new_vec->cords = build_vector_from_list(py_vec);
        if(new_vec->cords == NULL){
            free_vectors(head);
            return NULL;
        }
        new_vec->next = NULL;
        if (head == NULL) {
            head = new_vec;
            curr = head;
        } else {
            curr->next = new_vec;
            curr = curr->next;
        }
    }
    return head;
}

static struct cord *list_to_vectors(PyObject *py_list) {
    int i, d = PyObject_Length(py_list);
    struct cord *head = NULL, *curr = NULL, *new_cord = NULL;
    double val;
    PyObject* item = NULL;
    
    if (d <= 0) {
        return NULL;
    }

    for (i = 0; i < d; i++) {
        item = PyList_GetItem(py_list, i);
        if (item == NULL) {  
            free_cords(head);
            return NULL;
        }
        val = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {  
            free_cords(head);
            PyErr_Clear();
            return NULL;
        }
        new_cord = malloc(sizeof(struct cord));
        if (new_cord == NULL) {
            free_cords(head);
            return NULL;
        }
        new_cord->value = val;
        new_cord->next = NULL;
        if (head == NULL) {
            head = new_cord;
            curr = head;
        } else {
            curr->next = new_cord;
            curr = curr->next;
        }
    }
    return head;
}

static struct cord ** pylist_to_cords_matrix(PyObject *py_matrix, int n) {
    PyObject *py_cords_list = NULL;
    int i;
    struct cord **cords_matrix = calloc(n, sizeof(struct cord*));

    if (cords_matrix == NULL) {
        return NULL;
    }

    for (i = 0; i < n; i++) {
        py_cords_list = PyList_GetItem(py_matrix, i);
        if(py_cords_list == NULL) {
            free_cords_matrix(cords_matrix, i);
            return NULL;
        }
        cords_matrix[i] = list_to_vectors(py_cords_list);
        if (cords_matrix[i] == NULL) {
            free_cords_matrix(cords_matrix, i);
            return NULL;
        }
    }

    return cords_matrix;
}

static PyObject *vectors_to_pylist(struct cord *cords, int d) {
    int i;
    PyObject *list = NULL, *py_float = NULL;

    list = PyList_New(d);
    if (!list) return NULL;
    for (i = 0; i < d; ++i){
        py_float = PyFloat_FromDouble(cords->value);
        if (!py_float) {
            Py_DECREF(list);
            return NULL;
        }
        if (PyList_SetItem(list, i, py_float) < 0) {
            Py_DECREF(py_float);
            Py_DECREF(list);
            return NULL;
        }
        cords = cords->next;
    }
    return list;
}

static PyObject *matrix_to_pylist(struct cord **matrix, int n, int k) {
    PyObject *item = NULL, *res = NULL;
    int i;

    res = PyList_New(n);
    if (res == NULL){
        return NULL;
    }

    for (i = 0; i < n; i++) {
        item = vectors_to_pylist(matrix[i], k);
        if (!item) {
            Py_DECREF(res);
            return NULL;
        }
        if (PyList_SetItem(res, i, item) < 0) {
            Py_DECREF(item);
            Py_DECREF(res);
            return NULL;
        }
    }
    return res;
}

static PyObject *symnmf(PyObject *self, PyObject *args) {
    PyObject *py_Wmatrix = NULL, *py_Hmatrix = NULL, *res = NULL;
    int n, k;
    struct cord **Hmatrix = NULL;
    struct vector *Wmatrix = NULL;

    if (!PyArg_ParseTuple(args, "OO", &py_Hmatrix, &py_Wmatrix)) {
        return NULL;
    }

    n = PyObject_Length(py_Hmatrix);
    if (n <= 0) {
        return NULL;
    }

    Wmatrix = linkedList_to_vectors(py_Wmatrix);
    if (Wmatrix == NULL) {
        return NULL;
    }
    Hmatrix = pylist_to_cords_matrix(py_Hmatrix, n);
    if (Hmatrix == NULL) {
        free_vectors(Wmatrix);
        return NULL;
    }

    symnmf_symnmf(Hmatrix, Wmatrix);
    k = cords_len(Hmatrix[0]);
    res = matrix_to_pylist(Hmatrix, n, k);

    free_cords_matrix(Hmatrix, n);
    free_vectors(Wmatrix);

    return res;
}

static PyObject *sym(PyObject *self, PyObject *args) {
    PyObject *py_X_datapoints = NULL, *res = NULL;
    int n;
    struct cord **matrix = NULL;
    struct vector *X_datapoints = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_X_datapoints)) {
        return NULL;
    }
    n = PyObject_Length(py_X_datapoints);
    if (n <= 0) {
        return NULL;
    }

    X_datapoints = linkedList_to_vectors(py_X_datapoints);
    if (!X_datapoints) {
        return NULL;
    }

    matrix = calloc(n, sizeof(struct cord*));
    if (matrix == NULL) {
        free_vectors(X_datapoints);
        return NULL;
    }

    symnmf_sym(matrix, X_datapoints);
    res = matrix_to_pylist(matrix, n, n);

    free_cords_matrix(matrix, n);
    free_vectors(X_datapoints);

    return res;
}

static PyObject *ddg(PyObject *self, PyObject *args) {
    PyObject *py_X_datapoints = NULL, *res = NULL;
    int n;
    struct cord **matrix = NULL;
    struct vector *X_datapoints = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_X_datapoints)) {
        return NULL;
    }
    n = PyObject_Length(py_X_datapoints);
    if (n <= 0) {
        return NULL;
    }

    X_datapoints = linkedList_to_vectors(py_X_datapoints);
    if (!X_datapoints) {
        return NULL;
    }

    matrix = calloc(n, sizeof(struct cord*));
    if (matrix == NULL) {
        free_vectors(X_datapoints);
        return NULL;
    }

    symnmf_ddg(matrix, X_datapoints);
    res = matrix_to_pylist(matrix, n, n);

    free_cords_matrix(matrix, n);
    free_vectors(X_datapoints);

    return res;
}

static PyObject *norm(PyObject *self, PyObject *args) {
    PyObject *py_X_datapoints = NULL, *res = NULL;
    int n;
    struct cord **matrix = NULL;
    struct vector *X_datapoints = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_X_datapoints)) {
        return NULL;
    }
    n = PyObject_Length(py_X_datapoints);
    if (n <= 0) {
        return NULL;
    }

    X_datapoints = linkedList_to_vectors(py_X_datapoints);
    if (!X_datapoints) {
        return NULL;
    }

    matrix = calloc(n, sizeof(struct cord*));
    if (matrix == NULL) {
        free_vectors(X_datapoints);
        return NULL;
    }

    symnmf_norm(matrix, X_datapoints);
    res = matrix_to_pylist(matrix, n, n);

    free_cords_matrix(matrix, n);
    free_vectors(X_datapoints);

    return res;
}





static PyMethodDef symnmfMethods[] = {
    {
        "symnmf",  
        (PyCFunction)symnmf,          
        METH_VARARGS,
        PyDoc_STR("Perform full the symNMF as described")
    },{
        "sym",  
        (PyCFunction)sym,           
        METH_VARARGS,
        PyDoc_STR("Calculate and return the similarity matrix as described")
    },{
        "ddg",  
        (PyCFunction)ddg,  
        METH_VARARGS,
        PyDoc_STR("Calculate and return the Diagonal Degree Matrix as described")
    },{
        "norm",  
        (PyCFunction)norm,           
        METH_VARARGS,
        PyDoc_STR("Calculate and return the normalized similarity matrix as describedx")
    },{
        NULL, NULL, 0, NULL
    }
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule", /* name of module */
    NULL,
    -1,
    symnmfMethods, 
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    PyObject *projectmodule;
    projectmodule = PyModule_Create(&symnmfmodule);
    if (!projectmodule) {
        return NULL;
    }
    return projectmodule;
}