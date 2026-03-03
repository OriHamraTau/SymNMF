#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "symnmf.h"

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#define SMALL_NUM 1e-9
#define MAX_ITER 300
#define EPSILON 1e-4

void free_matrix(double **M, int n) {
    int i;
    if (M == NULL) return;
    for (i = 0; i < n; i += 1) free(M[i]);
    free(M);
}

static void free_matrices(double **WH, double **HHt, double **HHtH, int n) {
    free_matrix(WH, n);
    free_matrix(HHt, n);
    free_matrix(HHtH, n);
}

void free_cords(struct cord *head) {
    struct cord *tmp;
    while (head != NULL) {
        tmp = head;
        head = head->next;
        free(tmp);
    }
}

void free_vectors(struct vector *head) {
    struct vector *tmp;
    while (head != NULL) {
        tmp = head;
        head = head->next;
        free_cords(tmp->cords);
        free(tmp);
    }
}

static void *malloc_size_t(size_t count, size_t elem) {
    if (count != 0 && elem > (SIZE_MAX / count)) {
        return NULL;
    }
    return malloc(count * elem);
}

static void *calloc_size_t(size_t count, size_t elem) {
    if (count != 0 && elem > (SIZE_MAX / count)) {
        return NULL;
    }
    return calloc(count, elem);
}

int cords_len(struct cord *c) {
    int len = 0;
    while (c != NULL) {
        len++;
        c = c->next;
    }
    return len;
}

static double **alloc_dense(int n, int k) {
    int i, j;
    double **M = malloc_size_t((size_t)n, sizeof(*M));
    if (M == NULL) {
        return NULL;
    }

    for (i = 0; i < n; i++) {
        M[i] = calloc_size_t((size_t)k, sizeof(**M));
        if (M[i] == NULL) {
            free_matrix(M, i);
            return NULL;
        }
    }
    return M;
}

static int alloc_matrices(int n, int k, double ***WH, double ***HHt, double ***HHtH) {
    *WH = NULL;
    *HHt = NULL; 
    *HHtH = NULL;
    
    *WH  = alloc_dense(n, k);
    if (*WH == NULL) return -1;

    *HHt = alloc_dense(n, n);
    if (*HHt == NULL) {
        free_matrix(*WH, n); *WH = NULL;
        return -1; 
    }

    *HHtH = alloc_dense(n, k);
    if (*HHtH == NULL) {
        free_matrix(*WH,  n); *WH  = NULL;
        free_matrix(*HHt, n); *HHt = NULL;
        return -1;
    }
    return 0;
}

static int vectors_len(struct vector *v) {
    int len = 0;
    while (v != NULL) {
        len++;
        v = v->next;
    }
    return len;
}

static char *my_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = malloc(n);

    if (p == NULL){
        return NULL;
    }

    memcpy(p, s, n);
    return p;
}

struct cord *cords_from_array(const double *arr, int k) {
    int i;
    struct cord *head = NULL, *curr = NULL, *new_cord = NULL;

    for (i = 0; i < k; i++) {
        new_cord = malloc(sizeof(struct cord));
        if (new_cord == NULL) {
            free_cords(head);
            return NULL;
        }

        new_cord->next = NULL;
        new_cord->value = arr[i];

        if (head == NULL) {
            head = new_cord;
            curr = head;
        } else {
            curr->next = new_cord;
            curr = curr->next;
            curr->next = NULL;
        }
    }
    return head;
}

static struct cord *vectors_from_dense(const double **M, int n, int k) {
    int i;
    struct vector *head = NULL, *curr = NULL, *new_vec = NULL;
    
    for (i = 0; i < n; i++) {
        new_vec = malloc(sizeof(struct vector));
        if (new_vec == NULL) {
            printf("An Error Has Occurred\n");
            exit(1);
        }

        new_vec->next = NULL;
        new_vec->cords = cords_from_array(M[i], k);

        if (head == NULL) {
            head = new_vec;
            curr = head;
        } else {
            curr->next = new_vec;
            curr = curr->next;
            curr->next = NULL;
        }
    }
    return head;
}

static void dense_to_cord_rows(double **M, int n, int k, struct cord **rows_out) {
    int i;
    for (i = 0; i < n; i += 1) {
        rows_out[i] = cords_from_array(M[i], k);
    }
}

static double **dense_from_vectors(struct vector *V,  int *rows_out, int *cols_out) {
    int i, j, n, k;
    struct cord *curr_cord;
    struct vector *curr_vec;
    double **M;

    if (V == NULL) {
        *rows_out = 0;
        *cols_out = 0;
        return NULL;
    }

    n = vectors_len(V);
    k = cords_len(V->cords);
    M = alloc_dense(n,k);

    if (M == NULL) {
        *rows_out = 0;
        *cols_out = 0;
        return NULL;
    }

    curr_vec = V;
    for (i = 0; i < n && curr_vec != NULL; i++) {
        curr_cord = curr_vec->cords;
        for (j = 0; j < k && curr_cord != NULL; j++) {
            M[i][j] = curr_cord->value;
            curr_cord = curr_cord->next;
        }
        curr_vec = curr_vec->next;
    }

    *rows_out = n;
    *cols_out = k;
    return M;
}

/* WH = W (nxn) * H (nxk) */
static void compute_WH(double **W, double **H, double **WH, int n, int k) {
    int i, j, r;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++)
        {
            WH[i][j] = 0.0;
            for (r = 0; r < n; r++)
            {
                WH[i][j] += W[i][r] * H[r][j];
            }
        }
    }
}

/* HHt = H (nxk) * H^t (kxn) */
static void compute_HHt(double **H, double **HHt, int n, int k) {
    int i, j, r;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            HHt[i][j] = 0.0;
            for (r = 0; r < k; r++) {
                HHt[i][j] += H[i][r] * H[j][r];
            }
        }
    }
}

/* HHtH = (H * H^t) * H */
static void compute_HHtH(double **H, double **HHt, double **HHtH, int n, int k) {
    int i, j, r;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            HHtH[i][j] = 0.0;
            for (r = 0; r < n; r++) {
                HHtH[i][j] += HHt[i][r] * H[r][j];
            }
        }
    }
}

static double update_H_step(double **H, double **WH, double **HHtH, int n, int k) {
    int i, j;
    double old_val, new_val, diff_sum = 0.0, denom;
    const double beta = 0.5;

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            denom = HHtH[i][j];
            if (denom < SMALL_NUM) {
                denom = SMALL_NUM;
            }

            old_val = H[i][j];
            new_val = old_val * ((1.0 - beta) + beta * (WH[i][j] / denom));
            if (new_val < 0.0) {
                new_val = 0.0;
            }

            H[i][j] = new_val;
            diff_sum += (new_val - old_val) * (new_val - old_val);
        }
    }
    return diff_sum;
}


static void iter_symnmf(double **W, double **H,double **WH, double **HHt, double **HHtH,int n, int k) {
    int iter;
    double diff_sum;

    for (iter = 0; iter < MAX_ITER; iter++) {
        compute_WH(W, H, WH, n, k);
        compute_HHt(H, HHt, n, k);
        compute_HHtH(H, HHt, HHtH, n, k);
        diff_sum = update_H_step(H, WH, HHtH, n, k);
        if (diff_sum < EPSILON) break;
    }
}

static double **H_cords_to_dense(struct cord **H, int n, int k) {
    int i, j;
    struct cord *curr;
    double **Hmatrix;

    Hmatrix = alloc_dense(n,k);
    if(Hmatrix == NULL) {
        return NULL;
    }

    for (i = 0; i < n; i++) {
        curr = H[i];
        for (j = 0; j < k && curr != NULL; j++) {
            Hmatrix[i][j] = (curr ->value >= 0.0)? curr->value : 0.0;
            curr = curr->next;
        }
    }
    return Hmatrix;
}

void symnmf_symnmf(struct cord **H, struct vector *W) {
    int w_rows, w_cols, n, i, k;
    double **Wmatrix, **Hmatrix, **WH, **HHt, **HHtH;

    if (H == NULL || H[0] == NULL || W == NULL) {
        return;
    }

    k = cords_len(H[0]);
    if (k <= 0) {
        return;
    }

    Wmatrix = dense_from_vectors(W, &w_rows, &w_cols);
    if (w_rows == 0 || w_rows != w_cols) {
        free_matrix(Wmatrix, w_rows);
        return;
    }
    n = w_rows;
    Hmatrix = H_cords_to_dense(H, n, k);
    if (Hmatrix == NULL) { 
    free_matrix(Wmatrix, n); 
    return; 
    }
    if (alloc_matrices(n, k, &WH, &HHt, &HHtH) != 0) {
        free_matrix(Wmatrix, n);
        free_matrix(Hmatrix, n);
        return;
    }

    iter_symnmf(Wmatrix, Hmatrix, WH, HHt, HHtH, n, k);

    for (i = 0; i < n; i ++) {
        free_cords(H[i]);
        H[i] = NULL;
    }

    dense_to_cord_rows(Hmatrix, n, k, H);
    free_matrix(Hmatrix, n);
    free_matrix(Wmatrix, n);
    free_matrices(WH, HHt, HHtH, n);
}