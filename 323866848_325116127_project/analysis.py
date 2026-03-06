import user_input
import numpy as np
import kmeans
from sklearn.metrics import silhouette_score
import symnmfmodule


EPSILON = 0.0001
MAX_ITER = 300
Input_Length = 3

def main():
    user_input.validate_length_of_input(Input_Length)
    k, file_name = user_input.get_arguments(Input_Length)

    # load files into NumPy arrays
    data = np.loadtxt(file_name, delimiter=',', ndmin=2)
    # get dimensions
    n, d = data.shape

    user_input.validate_k(k, n)
    user_input.validate_file(file_name)

    vectors = data.tolist()
    kmeans_labels = kmeans_algo(k, vectors)
    symnmf_labels = symnmf_algo(data, k)

    # Print the silhouette scores of the symNMF and k-means algorithms.
    print_symnmf_score(data, symnmf_labels)
    print_kmeans_score(data, kmeans_labels)
    


def kmeans_algo(k, vectors):
    kmeans_labels = []
    initial_centroids = kmeans.initialize_centroids(k, vectors)
    vectors_per_centroid = kmeans.intialize_vectors_per_centroid(k)
    final_centroids = kmeans.k_means(initial_centroids, vectors, MAX_ITER, vectors_per_centroid)

    for vec in vectors:
        assigned_id = kmeans.get_centroid_id_per_vector(vec, final_centroids)
        kmeans_labels.append(assigned_id)
    
    return kmeans_labels
    
def print_kmeans_score(data, kmeans_labels):
    kmeans_score = silhouette_score(data, kmeans_labels)
    print("kmeans: {:.4f}".format(kmeans_score))


def symnmf_algo(data, k):
    W = symnmfmodule.norm(data.tolist())
    H_initial_matrix = user_input.H_initialization(W, k)
    H_final_matrix = symnmfmodule.symnmf(H_initial_matrix, W)

    return [row.index(max(row)) for row in H_final_matrix]

def print_symnmf_score(data, symnmf_labels):
    symnmf_score = silhouette_score(data, symnmf_labels)
    print("nmf: {:.4f}".format(symnmf_score))

if __name__ == "__main__":
    main()