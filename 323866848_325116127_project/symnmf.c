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
    int i;
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

static struct vector *vectors_from_dense(double **M, int n, int k) {
    int i;
    struct vector *head = NULL, *curr_vec = NULL, *new_vec = NULL;

    for (i = 0; i < n; i++) {
        new_vec = malloc(sizeof(struct vector));
        if (new_vec == NULL) {
            printf("An Error Has Occurred\n");
            exit(1);
        }
        new_vec->cords = cords_from_array(M[i], k);
        new_vec->next = NULL;
        if (head == NULL) {
            head = new_vec;
            curr_vec = new_vec;
        } else {
            curr_vec->next = new_vec;
            curr_vec = curr_vec->next;
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
    double w_val;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) WH[i][j] = 0.0;
        for (r = 0; r < n; r++) {
            w_val = W[i][r];
            for (j = 0; j < k; j++) {
                WH[i][j] += w_val * H[r][j];
            }
        }
    }
}

/* HHt = H (nxk) * H^t (kxn) */
static void compute_HHt(double **H, double **HHt, int n, int k) {
    int i, j, r;
    double sum;
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            sum = 0.0;
            for (r = 0; r < k; r++) {
                sum += H[i][r] * H[j][r];
            }
            HHt[i][j] = sum;
            HHt[j][i] = sum;
        }
    }
}

/* HHtH = (H * H^t) * H */
static void compute_HHtH(double **H, double **HHt, double **HHtH, int n, int k) {
    int i, j, r;
    double hht_val;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) HHtH[i][j] = 0.0;
        for (r = 0; r < n; r++) {
            hht_val = HHt[i][r];
            for (j = 0; j < k; j++) {
                HHtH[i][j] += hht_val * H[r][j];
            }
        }
    }
}

static double update_H_step(double **H, double **WH, double **HHtH, int n, int k) {
    int i, j;
    double denom, old_val, new_val, diff, diff_sum = 0.0;
    const double beta = 0.5;

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            old_val = H[i][j];
            denom = HHtH[i][j];
            if (denom < SMALL_NUM) denom = SMALL_NUM;

            new_val = old_val * ((1.0 - beta) + beta * (WH[i][j] / denom));
            if (new_val < 0.0) new_val = 0.0;

            H[i][j] = new_val;
            diff = new_val - old_val;
            diff_sum += diff * diff;
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



static void* safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        printf("An Error Has Occurred\n");
        exit(1);
    }
    return ptr;
}


static double euclidean_distance(double *a, double *b, int d) {
    double sum = 0.0, diff;
    int i;
    for (i = 0; i < d; i++) {
        diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

void fill_sym_dense(double **A, double **matrix_X, int n, int d) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            if (i == j) {
                A[i][j] = 0.0;
            } else {
                double dist = euclidean_distance(matrix_X[i], matrix_X[j], d);
                double val = exp(-dist / 2.0);
                A[i][j] = val;
                A[j][i] = val;
            }
        }
    }
}

/* Calculate and output the similarity matrix */
void symnmf_sym(struct cord **out_rows, struct vector *data_rows){
    int n, d, i, j;
    double **X_dense, **A_dense;

    if (data_rows == NULL) return;

    n = vectors_len(data_rows);
    d = cords_len(data_rows->cords);

    X_dense = dense_from_vectors(data_rows, &i, &j); 
    A_dense = alloc_dense(n, n);

    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            if (i == j) {
                A_dense[i][j] = 0.0;
            } else {
                double dist = euclidean_distance(X_dense[i], X_dense[j], d);
                double val = exp(-dist / 2.0);
                A_dense[i][j] = val;
                A_dense[j][i] = val;
            }
        }
    }
    dense_to_cord_rows(A_dense, n, n, out_rows);
    free_matrix(X_dense, n);
    free_matrix(A_dense, n);
}

static double* get_diagonal(double **matrix_X, int n, int d) {
    int i, j;
    double dist;
    double *diagonal = (double*)safe_malloc(sizeof(double) * n);

    for (i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (j = 0; j < n; j++) {
            if (i != j) {
                dist = euclidean_distance(matrix_X[i], matrix_X[j], d);
                row_sum += exp(-dist / 2.0);
            }
        }
        diagonal[i] = row_sum;
    }
    return diagonal;
}

/* Calculate and output the Diagonal Degree Matrix */
void symnmf_ddg(struct cord **out_rows, struct vector *data_rows) {
    int n, d, i, j;
    double **X_dense, *diagonal;
    double **D_matrix;

    n = vectors_len(data_rows);
    d = cords_len(data_rows->cords);

    X_dense = dense_from_vectors(data_rows, &i, &j);
    diagonal = get_diagonal(X_dense, n, d);
    D_matrix = alloc_dense(n, n);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            D_matrix[i][j] = (i == j) ? diagonal[i] : 0.0;
        }
    }

    dense_to_cord_rows(D_matrix, n, n, out_rows);

    free_matrix(X_dense, n);
    free_matrix(D_matrix, n);
    free(diagonal);
}

/* Calculate and output the normalized similarity matrix */
void symnmf_norm(struct cord **out_rows, struct vector *data_rows) {
    int n, d, i, j;
    double **X_dense, **A_dense, *D_diag;

    n = vectors_len(data_rows);
    d = cords_len(data_rows->cords);

    X_dense = dense_from_vectors(data_rows, &i, &j);
    A_dense = alloc_dense(n, n);
    D_diag = (double*)calloc(n, sizeof(double));

    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            double dist = euclidean_distance(X_dense[i], X_dense[j], d);
            double val = exp(-dist / 2.0);
            A_dense[i][j] = val;
            A_dense[j][i] = val;
            D_diag[i] += val;
            D_diag[j] += val;
        }
    }

    for (i = 0; i < n; i++) {
        double d_i = (D_diag[i] > 1e-12) ? 1.0 / sqrt(D_diag[i]) : 0.0;
        for (j = 0; j < n; j++) {
            double d_j = (D_diag[j] > 1e-12) ? 1.0 / sqrt(D_diag[j]) : 0.0;
            A_dense[i][j] *= (d_i * d_j);
        }
    }

    dense_to_cord_rows(A_dense, n, n, out_rows);

    free_matrix(A_dense, n);
    free_matrix(X_dense, n);
    free(D_diag);
}


/* ------------ CLI Implementaion ------------ */

static char *my_strdup(const char *str) {
    size_t n = strlen(str) + 1;
    char *p = malloc(n);
    if (p == NULL) return NULL;
    memcpy(p, str, n);
    return p;
}

static int is_whitespace(const char *str) {
    const char *p = str;

    while (*p != '\0') {
        if (!(*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) return 0;
        p++;
    }
    return 1;
}

static int validate_cap(char **buf, size_t *cap, size_t need) {
    size_t new_cap;
    char *tmp;

    if (need <= *cap) return 0;

    new_cap = (*cap == 0) ? 512 : *cap;
    while (new_cap < need) {
        if (new_cap > SIZE_MAX / 2) {
            new_cap = need;
            if (new_cap < *cap) return -1;
            break;
        }
        new_cap *= 2;
    }

    tmp = (char *)realloc(*buf, new_cap);
    if (!tmp) return -1;

    *buf = tmp;
    *cap = new_cap;
    return 0;
}

static char *readline_dynamic(FILE *f) {
    char *buffer = NULL;
    size_t cap = 0, len = 0;
    char chunk[4096];

    for (;;) {
        if (fgets(chunk, (int)sizeof(chunk), f) == NULL) {
            if (len == 0) { free(buffer); return NULL; }
            if (validate_cap(&buffer, &cap, len + 1) != 0) { free(buffer); return NULL; }
            buffer[len] = '\0';
            return buffer;
        }

        {
            size_t tlen = strlen(chunk);
            if (validate_cap(&buffer, &cap, len + tlen + 1) != 0) { free(buffer); return NULL; }
            memcpy(buffer + len, chunk, tlen);
            len += tlen;
            buffer[len] = '\0';
        }

        if (len > 0 && buffer[len - 1] == '\n') return buffer;
    }
}

static int is_txt(const char *path) {
    const char *dot = strrchr(path, '.');
    if (!dot) return 0;
    if (dot == path) return 0;
    return (dot[1] == 't' && dot[2] == 'x' && dot[3] == 't' && dot[4] == '\0');
}

static int columns_count(const char *str) {
    const char *p;
    int cnt = 1;
    for (p = str; *p != '\0'; p += 1) if (*p == ','){
        cnt ++;
    }
    return cnt;
}

static int grow_matrix(double ***M, int *cap) {
    double **tmp;
    int old = *cap;

    if ((size_t)(*cap) > SIZE_MAX / (2 * sizeof(*tmp))) {
        return -1;
    }

    *cap *= 2;
    tmp = realloc(*M, (size_t)(*cap) * sizeof(*tmp));
    if (tmp == NULL) { 
        *cap = old; 
        return -1;
    }
    *M = tmp;
    return 0;
}

static int parse_txt_row(const char *line_in, double **out_row, int m) {
    char *tok, *endp, *line = my_strdup(line_in);
    int j;
    double v;

    if (line == NULL) return -1;

    *out_row = malloc_size_t((size_t)m, sizeof(**out_row));
    if (*out_row == NULL) { free(line); return -1; }

    tok = strtok(line, ",\r\n");
    j = 0;
    while (tok != NULL && j < m) {
        v = strtod(tok, &endp);

        while (*endp == ' ' || *endp == '\t') endp++;

        if (endp == tok || *endp != '\0') {
            free(*out_row);
            free(line);
            return -1;
        }

        (*out_row)[j] = v;
        j += 1;
        tok = strtok(NULL, ",\r\n");
    }

    if (j != m) { free(*out_row); free(line); return -1; }
    free(line);
    return 0;
}

static int read_txt(const char *path, double ***out_M, int *out_n, int *out_k) {
    FILE *f;
    int cap, n, k, cols;
    double **M;
    char *line;
    if (!is_txt(path)) return -1;
    cap = 128; n = 0; k = -1;

    f = fopen(path, "r");
    if (f == NULL) return -1;

    M = malloc_size_t((size_t)cap, sizeof(*M));
    if (M == NULL) {
        fclose(f);
        return -1;
    }

    while ((line = readline_dynamic(f)) != NULL) {
        if (is_whitespace(line)) {
             free(line); free_matrix(M, n); fclose(f); return -1; }
        
        cols = columns_count(line);
        if (k < 0) {
            k = cols;
        } else if (cols != k) {
            free(line); free_matrix(M, n); fclose(f); return -1;}
        
        if (parse_txt_row(line, &M[n], k) != 0) {
            free(line); free_matrix(M, n); fclose(f); return -1;
        }
        free(line);
        n += 1;
        if (n == cap && grow_matrix(&M, &cap) != 0) {
            free_matrix(M, n); fclose(f); return -1;
        }
    }
    fclose(f);
    if (k <= 0 || n <= 0) { free_matrix(M, n); return -1; }
    *out_M = M;
    *out_n = n;
    *out_k = k;
    return 0;
}

static int load_input(const char *path, struct vector **out_vecs,
                        double ***out_dense, int *out_n, int *out_d) {
    double **X_dense;
    int n, d;
    struct vector *X_vecs;

    if (read_txt(path, &X_dense, &n, &d) != 0) return -1;

    X_vecs = vectors_from_dense(X_dense, n, d);
    *out_vecs = X_vecs;
    *out_dense = X_dense;
    *out_n = n;
    *out_d = d;
    return 0;
}

static void print_rows(struct cord **rows, int n, int k) {
    int i = 0, j = 0;
    struct cord *curr = NULL;

    for (i = 0; i < n; i++) {
        curr = rows[i];
        for (j = 0; j < k; j++) {
            if (curr == NULL) {
                printf("An Error Has Occurred\n");
                exit(1);
            }
            if (j == k - 1) {
                printf("%.4f\n", curr->value);
            } else {
                printf("%.4f,", curr->value);
            }
            curr = curr->next;
        }
    }
}

static void execute_and_print(const char *goal, struct vector *X_vecs, int n) {
    int i;
    struct cord **rows;
    if (strcmp(goal, "symnmf") == 0) {
        printf("An Error Has Occurred\n");
        exit(1);
    }
    rows = (struct cord **)calloc_size_t((size_t)n, sizeof(*rows));
    if (rows == NULL) {
        printf("An Error Has Occurred\n");
        exit(1);
    }
    if (strcmp(goal, "sym") == 0) symnmf_sym(rows, X_vecs);
    else if (strcmp(goal, "ddg") == 0) symnmf_ddg(rows, X_vecs);
    else if (strcmp(goal, "norm") == 0) symnmf_norm(rows, X_vecs);
    else {
        free(rows);
        printf("An Error Has Occurred\n");
        exit(1); 
    }
    print_rows(rows, n, n);
    for (i = 0; i < n; i++) free_cords(rows[i]);
    free(rows);
}

int main(int argc, char **argv) {
    const char *goal, *path;
    int n, d;
    struct vector *X_vecs;
    double **X_dense;

    if (argc != 3) {
        printf("An Error Has Occurred\n");
        exit(1);
    }
    goal = argv[1];
    path = argv[2];

    X_vecs = NULL;
    X_dense = NULL;

    if (load_input(path, &X_vecs, &X_dense, &n, &d) != 0) {
        printf("An Error Has Occurred\n");
        exit(1);
    }
    execute_and_print(goal, X_vecs, n);

    free_vectors(X_vecs);
    if (X_dense != NULL) free_matrix(X_dense, n);
    return 0;
}
