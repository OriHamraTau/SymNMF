#ifndef SYMNMF_H
#define SYMNMF_H

/* data structures */

struct cord {
    double value;
    struct cord *next;
};

struct vector {
    struct cord *cords;
    struct vector *next;
};

/* exported helpers */

int cords_len(struct cord *c);
void free_cords(struct cord *head);
void free_vectors(struct vector *head);
void free_matrix(double **M, int n);
struct cord *cords_from_array(const double *arr, int m);

/* exported algorithms */

// Perform full the symNMF as described
void symnmf_symnmf(struct cord **H, struct vector *W);

// Calculate and output the similarity matrix
void symnmf_sym(struct cord **out_rows, struct vector *data_rows);

// Calculate and output the Diagonal Degree Matrix
void symnmf_ddg(struct cord **out_rows, struct vector *data_rows);

// Calculate and output the normalized similarity matrix
void symnmf_norm(struct cord **out_rows, struct vector *data_rows);

#endif /* SYMNMF_H */
