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

/**
 * @brief Frees a dynamically allocated 2D matrix.
 *
 * @param M The matrix to free.
 * @param n The number of rows in the matrix.
 */
void free_matrix(double **M, int n) {
    int i;
    if (M == NULL) return;
    for (i = 0; i < n; i += 1) free(M[i]);
    free(M);
}

/**
 * @brief Frees multiple matrices used in the algorithm at once.
 *
 * @param WH The W*H matrix to free.
 * @param HHt The H*H^T matrix to free.
 * @param HHtH The H*H^T*H matrix to free.
 * @param n The number of rows in each matrix.
 */
static void free_matrices(double **WH, double **HHt, double **HHtH, int n) {
    free_matrix(WH, n);
    free_matrix(HHt, n);
    free_matrix(HHtH, n);
}

/**
 * @brief Frees a linked list of coordinates.
 *
 * @param head Pointer to the head of the linked list.
 */
void free_cords(struct cord *head) {
    struct cord *tmp;
    while (head != NULL) {
        tmp = head;
        head = head->next;
        free(tmp);
    }
}

/**
 * @brief Frees a linked list of vectors and their inner coordinates.
 *
 * @param head Pointer to the head of the vector linked list.
 */
void free_vectors(struct vector *head) {
    struct vector *tmp;
    while (head != NULL) {
        tmp = head;
        head = head->next;
        free_cords(tmp->cords);
        free(tmp);
    }
}

/**
 * @brief Safely allocates memory using malloc, protecting against integer overflow.
 *
 * @param count Number of elements to allocate.
 * @param elem Size of each element in bytes.
 * @return void* Pointer to the allocated memory, or NULL on overflow/failure.
 */
static void *malloc_size_t(size_t count, size_t elem) {
    if (count != 0 && elem > (SIZE_MAX / count)) {
        return NULL;
    }
    return malloc(count * elem);
}

/**
 * @brief Safely allocates zero-initialized memory using calloc, protecting against overflow.
 *
 * @param count Number of elements to allocate.
 * @param elem Size of each element in bytes.
 * @return void* Pointer to the allocated memory, or NULL on overflow/failure.
 */
static void *calloc_size_t(size_t count, size_t elem) {
    if (count != 0 && elem > (SIZE_MAX / count)) {
        return NULL;
    }
    return calloc(count, elem);
}

/**
 * @brief Calculates the length (number of elements) of a coordinate linked list.
 *
 * @param c Pointer to the head of the coordinate linked list.
 * @return int The number of coordinates in the list.
 */
int cords_len(struct cord *c) {
    int len = 0;
    while (c != NULL) {
        len++;
        c = c->next;
    }
    return len;
}

/**
 * @brief Calculates the length (number of elements) of a vector linked list.
 *
 * @param v Pointer to the head of the vector linked list.
 * @return int The number of vectors in the list.
 */
static int vectors_len(struct vector *v) {
    int len = 0;
    while (v != NULL) {
        len++;
        v = v->next;
    }
    return len;
}

/**
 * @brief Allocates a dense 2D matrix of doubles.
 *
 * The function allocates an n x k matrix, initializing all values to zero.
 * It also handles partial allocation failures safely.
 *
 * @param n Number of rows.
 * @param k Number of columns.
 * @return double** Pointer to the allocated matrix, or NULL if allocation fails.
 */
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

/**
 * @brief Allocates the three main matrices used in the SymNMF algorithm.
 *
 * Safely allocates WH, HHt, and HHtH matrices. If any allocation fails,
 * it frees previously allocated memory to prevent leaks.
 *
 * @param n Number of data points.
 * @param k Number of clusters.
 * @param WH Pointer to store the allocated W*H matrix (n x k).
 * @param HHt Pointer to store the allocated H*H^T matrix (n x n).
 * @param HHtH Pointer to store the allocated H*H^T*H matrix (n x k).
 * @return int 0 on success, -1 if an allocation failure occurred.
 */
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

/**
 * @brief Creates a linked list of coordinates from an array of doubles.
 *
 * @param arr The array containing the coordinate values.
 * @param k The number of elements in the array.
 * @return struct cord* Pointer to the head of the new coordinate list, or NULL on failure.
 */
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

/**
 * @brief Converts a dense 2D matrix into a linked list of vectors.
 *
 * @param M The 2D array (matrix) of data points.
 * @param n The number of rows (vectors) in the matrix.
 * @param k The number of columns (coordinates per vector).
 * @return struct vector* Pointer to the head of the new vector list.
 */
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

/**
 * @brief Converts a 2D dense matrix into an array of coordinate linked lists.
 *
 * @param M The dense 2D matrix.
 * @param n Number of rows in the matrix.
 * @param k Number of columns (coordinates) in the matrix.
 * @param rows_out Array of struct cord pointers to store the resulting linked lists.
 */
static void dense_to_cord_rows(double **M, int n, int k, struct cord **rows_out) {
    int i;
    for (i = 0; i < n; i += 1) {
        rows_out[i] = cords_from_array(M[i], k);
    }
}

/**
 * @brief Converts a linked list of vectors into a dynamically allocated 2D matrix.
 *
 * @param V Pointer to the head of the vector linked list.
 * @param rows_out Pointer to an int where the number of rows (vectors) will be stored.
 * @param cols_out Pointer to an int where the number of columns (coordinates) will be stored.
 * @return double** Pointer to the allocated 2D matrix, or NULL if allocation fails.
 */
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

/**
 * @brief Computes the matrix multiplication of W and H.
 *
 * @param W The n x n matrix W (similarity or normalized matrix).
 * @param H The n x k matrix H.
 * @param WH The output n x k matrix to store the result (W * H).
 * @param n Number of rows/cols in W and rows in H.
 * @param k Number of columns in H.
 */
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

/**
 * @brief Computes the matrix multiplication of H and H^T.
 *
 * This function exploits the symmetry of the resulting H*H^T matrix
 * to optimize calculations and reduce the number of iterations.
 *
 * @param H The n x k matrix H.
 * @param HHt The output n x n symmetric matrix to store the result (H * H^T).
 * @param n Number of rows in H.
 * @param k Number of columns in H.
 */
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

/**
 * @brief Computes the matrix multiplication of (H * H^T) and H.
 *
 * @param H The n x k matrix H.
 * @param HHt The n x n matrix representing H * H^T.
 * @param HHtH The output n x k matrix to store the result ((H * H^T) * H).
 * @param n Number of rows in H and HHt.
 * @param k Number of columns in H.
 */
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

/**
 * @brief Performs a single update step for the H matrix.
 *
 * Calculates the new values for H based on the SymNMF update rule,
 * using a beta parameter of 0.5. It also calculates the squared 
 * Frobenius norm of the difference between the old and new H.
 *
 * @param H The n x k matrix to be updated (in place).
 * @param WH The precomputed W * H matrix.
 * @param HHtH The precomputed H * H^T * H matrix.
 * @param n Number of rows in H.
 * @param k Number of columns in H.
 * @return double The sum of squared differences (diff_sum) for convergence checking.
 */
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

/**
 * @brief Executes the main iterative optimization loop for SymNMF.
 *
 * Iteratively updates the H matrix until the difference between 
 * consecutive iterations falls below EPSILON, or MAX_ITER is reached.
 *
 * @param W The normalized similarity matrix W (n x n).
 * @param H The initial H matrix (n x k), which gets updated to the final result.
 * @param WH Workspace matrix for W * H.
 * @param HHt Workspace matrix for H * H^T.
 * @param HHtH Workspace matrix for H * H^T * H.
 * @param n Number of data points.
 * @param k Number of clusters.
 */
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

/**
 * @brief Converts an array of coordinate linked lists into a dense 2D matrix, ensuring non-negative values.
 *
 * @param H Array of pointers to coordinate linked lists representing the H matrix.
 * @param n Number of rows (data points).
 * @param k Number of columns (clusters).
 * @return double** Pointer to the allocated 2D matrix, or NULL on allocation failure.
 */
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

/** Executes SymNMF algorithm: Wmatrix * Hmatrix iterations. Updates H in-place. */
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


/**
 * @brief Allocates memory safely and exits the program if allocation fails.
 *
 * @param size The number of bytes to allocate.
 * @return void* Pointer to the allocated memory.
 */
static void* safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        printf("An Error Has Occurred\n");
        exit(1);
    }
    return ptr;
}

/**
 * @brief Calculates the squared Euclidean distance between two vectors.
 *
 * @param a First vector.
 * @param b Second vector.
 * @param d The dimension of the vectors.
 * @return double The squared Euclidean distance.
 */
static double euclidean_distance(double *a, double *b, int d) {
    double sum = 0.0, diff;
    int i;
    for (i = 0; i < d; i++) {
        diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/**
 * @brief Fills the symmetric similarity matrix A using the Gaussian kernel.
 *
 * The diagonal elements are set to 0.0. For other elements, it calculates
 * the similarity using the formula: exp(-||x_i - x_j||^2 / 2).
 *
 * @param A The n x n symmetric similarity matrix to be filled.
 * @param matrix_X The n x d matrix containing the data points.
 * @param n Number of data points.
 * @param d Number of dimensions per data point.
 */
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


/**
 * @brief Allocates and builds the symmetric similarity matrix A.
 *
 * Computes the similarities between all pairs of points using the Gaussian kernel.
 *
 * @param X_dense The dense n x d matrix of data points.
 * @param n Number of data points.
 * @param d Number of dimensions per point.
 * @return double** Pointer to the allocated n x n similarity matrix.
 */
static double** build_similarity_matrix(double **X_dense, int n, int d) {
    int i, j;
    double **A_dense = alloc_dense(n, n);

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
    return A_dense;
}

/**
 * @brief Computes the Similarity Matrix (sym) from linked list input.
 *
 * @param out_rows Pointer to the array where the resulting rows will be stored as linked lists.
 * @param data_rows Linked list of input vectors.
 */
void symnmf_sym(struct cord **out_rows, struct vector *data_rows) {
    int n, d, dummy_r, dummy_c;
    double **X_dense, **A_dense;

    if (data_rows == NULL) return;

    n = vectors_len(data_rows);
    d = cords_len(data_rows->cords);

    X_dense = dense_from_vectors(data_rows, &dummy_r, &dummy_c); 
    
    A_dense = build_similarity_matrix(X_dense, n, d);
    
    dense_to_cord_rows(A_dense, n, n, out_rows);
    free_matrix(X_dense, n);
    free_matrix(A_dense, n);
}

/**
 * @brief Computes the diagonal values of the Diagonal Degree Matrix (D).
 *
 * The degree of a vertex is the sum of its similarities to all other vertices.
 *
 * @param matrix_X The dense n x d matrix of data points.
 * @param n Number of data points.
 * @param d Number of dimensions per point.
 * @return double* Array of size n containing the diagonal values.
 */
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

/**
 * @brief Computes the Diagonal Degree Matrix (ddg) from linked list input.
 *
 * @param out_rows Pointer to the array where the resulting D matrix will be stored as linked lists.
 * @param data_rows Linked list of input vectors.
 */
void symnmf_ddg(struct cord **out_rows, struct vector *data_rows) {
    int n, d, i, j, dummy_r, dummy_c;
    double **X_dense, *diagonal;
    double **D_matrix;

    n = vectors_len(data_rows);
    d = cords_len(data_rows->cords);

    X_dense = dense_from_vectors(data_rows, &dummy_r, &dummy_c);
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

/** Computes normalized similarity matrix W = D^-1/2 * A * D^-1/2. */
void symnmf_norm(struct cord **out_rows, struct vector *data_rows) {
    int n, d, i, j, dummy_r, dummy_c;
    double **X_dense, **A_dense, *D_diag;

    if (data_rows == NULL) return;

    n = vectors_len(data_rows);
    d = cords_len(data_rows->cords);

    X_dense = dense_from_vectors(data_rows, &dummy_r, &dummy_c);
    A_dense = build_similarity_matrix(X_dense, n, d);
    D_diag = (double*)calloc(n, sizeof(double));

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            D_diag[i] += A_dense[i][j];
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

/**
 * @brief Duplicates a string by allocating memory and copying its contents.
 *
 * @param str The string to duplicate.
 * @return char* Pointer to the newly allocated string, or NULL on failure.
 */
static char *my_strdup(const char *str) {
    size_t n = strlen(str) + 1;
    char *p = malloc(n);
    if (p == NULL) return NULL;
    memcpy(p, str, n);
    return p;
}

/**
 * @brief Checks if a string consists entirely of whitespace characters.
 *
 * @param str The string to check.
 * @return int 1 if the string is all whitespace, 0 otherwise.
 */
static int is_whitespace(const char *str) {
    const char *p = str;
    while (*p != '\0') {
        if (!(*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) return 0;
        p++;
    }
    return 1;
}

/**
 * @brief Ensures a dynamic buffer has enough capacity, reallocating if necessary.
 *
 * @param buf Pointer to the buffer.
 * @param cap Pointer to the current capacity of the buffer.
 * @param need The required capacity.
 * @return int 0 on success, -1 on allocation failure.
 */
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

/**
 * @brief Reads a line of arbitrary length from a file dynamically.
 *
 * @param f The file pointer to read from.
 * @return char* Pointer to the allocated string containing the line, or NULL on EOF/error.
 */
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

/**
 * @brief Checks if a given file path ends with the ".txt" extension.
 *
 * @param path The file path string.
 * @return int 1 if the extension is .txt, 0 otherwise.
 */
static int is_txt(const char *path) {
    const char *dot = strrchr(path, '.');
    if (!dot) return 0;
    if (dot == path) return 0;
    return (dot[1] == 't' && dot[2] == 'x' && dot[3] == 't' && dot[4] == '\0');
}

/**
 * @brief Counts the number of columns in a CSV string based on commas.
 *
 * @param str The comma-separated string.
 * @return int The number of columns.
 */
static int columns_count(const char *str) {
    const char *p;
    int cnt = 1;
    for (p = str; *p != '\0'; p += 1) if (*p == ','){
        cnt ++;
    }
    return cnt;
}

/**
 * @brief Doubles the capacity of a dynamically allocated array of double pointers.
 *
 * @param M Pointer to the array of double pointers.
 * @param cap Pointer to the current capacity (updated on success).
 * @return int 0 on success, -1 on allocation failure.
 */
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

/** Parses CSV string into double array. Returns 0 on success, -1 on error. */
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

/** Read CSV file into a dense 2D array. Returns 0 on success. */
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
             free(line); free_matrix(M, n); fclose(f); return -1; 
            }
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

/**
 * @brief High-level wrapper to load input data into both dense and linked list formats.
 *
 * @param path The file path to read from.
 * @param out_vecs Pointer to store the resulting linked list of vectors.
 * @param out_dense Pointer to store the resulting dense 2D matrix.
 * @param out_n Pointer to store the number of data points (rows).
 * @param out_d Pointer to store the dimensionality (columns).
 * @return int 0 on success, -1 on failure.
 */
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

/**
 * @brief Prints a list of coordinates as comma-separated values to 4 decimal places.
 *
 * @param rows Array of coordinate linked lists to print.
 * @param n Number of rows.
 * @param k Number of columns per row.
 */
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

/**
 * @brief Executes the requested matrix computation goal and prints the result.
 *
 * @param goal The string identifier of the goal ("sym", "ddg", or "norm").
 * @param X_vecs The input data formatted as a linked list of vectors.
 * @param n The number of data points.
 * @return  Return 0 on success, -1 on error.
 */
static int execute_and_print(const char *goal, struct vector *X_vecs, int n) {
    int i;
    struct cord **rows;
    if (strcmp(goal, "symnmf") == 0) {
        return -1;
    }
    rows = (struct cord **)calloc_size_t((size_t)n, sizeof(*rows));
    if (rows == NULL) {
        return -1;
    }
    if (strcmp(goal, "sym") == 0) symnmf_sym(rows, X_vecs);
    else if (strcmp(goal, "ddg") == 0) symnmf_ddg(rows, X_vecs);
    else if (strcmp(goal, "norm") == 0) symnmf_norm(rows, X_vecs);
    else {
        free(rows);
        return -1;
    }
    print_rows(rows, n, n);
    for (i = 0; i < n; i++) free_cords(rows[i]);
    free(rows);
    return 0;
}

int main(int argc, char **argv) {
    const char *goal, *path;
    int n, d, status = 0;
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
        status = 1;
    }
    if(execute_and_print(goal, X_vecs, n) != 0) {
        printf("An Error Has Occurred\n");
        status = 1;
    }

    free_vectors(X_vecs);
    if (X_dense != NULL) free_matrix(X_dense, n);

    if (status == 1) {
        exit(1);
    }
    return 0;
}
