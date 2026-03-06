#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "symnmf.h"

static int vectors_len(struct vector *v) {
    int len = 0;
    while (v != NULL) {
        len++;
        v = v->next;
    }
    return len;
}

static void* safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        printf("An Error Has Occurred\n");
        exit(1);
    }
    return ptr;
}

static void insert_cords(struct cord **head, struct cord **prev, struct cord *curr_node){ 
    if(*head == NULL){
        *head = curr_node;
        *prev = *head;
    }
    else{
        (*prev) -> next = curr_node; 
        *prev = curr_node;
    }
}

void free_cords(struct cord *head) {
    struct cord *tmp;
    while (head != NULL) {
        tmp = head;
        head = head->next;
        free(tmp);
    }
}

int cords_len(struct cord *c){
    int len = 0;
    while(c != NULL){
        len ++;
        c = c -> next;
    }
    return len;
}

static double euclidean_distance(struct vector *vec1, struct vector *vec2){
    double diff;
    double distance = 0.0;

    struct cord *cord1 = vec1 -> cords;
    struct cord *cord2 = vec2 -> cords;

    while(cord1 != NULL){
        diff = (cord1 -> value) - (cord2 -> value);
        distance += diff * diff;
        
        cord1 = cord1 -> next;
        cord2 = cord2 -> next;
    }
    return distance;
}

/* Calculate and output the similarity matrix */
void symnmf_sym(struct cord **out_rows, struct vector *data_rows){
    struct vector *curr_vec = data_rows;
    struct vector *second_vec;
    struct cord *head, *prev, *curr_dist;
    double distance = 0.0;
    int i = 0, j;

    while(curr_vec != NULL){
        second_vec = data_rows;
        head = NULL;
        prev = NULL;
        j = 0;

        while(second_vec != NULL){
            distance = euclidean_distance(curr_vec, second_vec);
            curr_dist = (struct cord*)safe_malloc(sizeof(struct cord));

            curr_dist->value = (i == j)? 0.0 : exp(-distance / 2.0);
            curr_dist -> next = NULL;
           
            insert_cords(&head, &prev, curr_dist);
            second_vec = second_vec -> next;
            j++;
        }
        out_rows[i] = head;
        curr_vec = curr_vec->next;
        i++;
    }
}

static double* get_diagonal(struct vector *data_rows, int N){
    int i;
    struct cord **A, *cur_node;
    double row_sum;

    A = (struct cord**)safe_malloc(sizeof(struct cord *) * N);
    symnmf_sym(A, data_rows);

    double* diagonal = (double*)safe_malloc(sizeof(double) * N);

    for (i = 0; i < N ; i++){
        row_sum = 0.0;
        cur_node = A[i]; /* run on each row */

        while(cur_node != NULL){
            row_sum += cur_node -> value;
            cur_node = cur_node -> next;
        }
        diagonal[i] = row_sum;
    }
    for(i = 0; i < N; i++){
        free_cords(A[i]);
    }
    free(A);
    return diagonal;
}

/* Calculate and output the Diagonal Degree Matrix */
void symnmf_ddg(struct cord **out_rows, struct vector *data_rows){
    int N, i, j;
    double row_sum;
    struct cord *head, *prev, *curr_row;

    N = vectors_len(data_rows);
    double* diagonal = get_diagonal(data_rows, N);

    for (i = 0; i < N ; i++){
        head = NULL;
        prev = NULL;
        row_sum = diagonal[i];

        for (j = 0; j < N; j++) {
            curr_row = (struct cord*)safe_malloc(sizeof(struct cord));

            curr_row -> value = (i == j)? row_sum : 0.0;
            curr_row -> next = NULL;

            insert_cords(&head, &prev, curr_row);
        }
        out_rows[i] = head;
    }
    free(diagonal);
}

/* Calculate and output the normalized similarity matrix */
void symnmf_norm(struct cord **out_rows, struct vector *data_rows){
    struct cord **A, *curr_cord, *row_i, *head, *prev;
    double *diagonal_root;
    int i, j, N;

    N = vectors_len(data_rows);
    A = (struct cord**)safe_malloc(sizeof(struct cord *) * N);
   
    diagonal_root = get_diagonal(data_rows, N);
    for(i = 0; i < N; i++){
        diagonal_root[i] = 1.00 / sqrt(diagonal_root[i]);
    }

    symnmf_sym(A, data_rows);

    for(i = 0; i < N; i++){
        head = NULL;
        prev = NULL;
        curr_cord = A[i];

        for(j = 0; j < N; j++){
            cord_j_in_row_i = (struct cord*)safe_malloc(sizeof(struct cord));

            cord_j_in_row_i -> value = diagonal_root[i] * (curr_cord -> value) * diagonal_root[j];
            cord_j_in_row_i -> next = NULL;
            
            insert_cords(&head, &prev, row_i);
            curr_cord = curr_cord -> next;
        }
        out_rows[i] = head;
    }

    for(i = 0; i < N; i++){
        free_cords(A[i]);
    }
    free(A);
    free(diagonal_root);
}