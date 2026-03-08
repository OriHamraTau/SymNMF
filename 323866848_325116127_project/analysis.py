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
    """
    Executes the K-means algorithm to cluster the given data points.
    
    This function initializes the centroids, runs the iterative K-means 
    algorithm to find the final centroids, and then assigns each input 
    vector to its closest final centroid.

    Args:
        k (int): The number of clusters to form.
        vectors (list): A list of data points (vectors) to be clustered.

    Returns:
        list: A list of integers representing the cluster labels (IDs) 
              assigned to each corresponding vector in the input.
    """
    kmeans_labels = []
    initial_centroids = kmeans.initialize_centroids(k, vectors)
    vectors_per_centroid = kmeans.initialize_vectors_per_centroid(k)
    final_centroids = kmeans.k_means(initial_centroids, vectors, MAX_ITER, vectors_per_centroid)

    for vec in vectors:
        assigned_id = kmeans.get_centroid_id_per_vector(vec, final_centroids)
        kmeans_labels.append(assigned_id)
    
    return kmeans_labels
    
def print_kmeans_score(data, kmeans_labels):
    """
    Calculates and prints the silhouette score for the K-means clustering results.

    Args:
        data (list or numpy.ndarray): The original data points.
        kmeans_labels (list): The cluster labels assigned to each data point by K-means.
    """
    kmeans_score = silhouette_score(data, kmeans_labels)
    print("kmeans: {:.4f}".format(kmeans_score))


def symnmf_algo(data, k):
    """
    Executes the Symmetric Non-negative Matrix Factorization (SymNMF) algorithm.

    This function calculates the normalized similarity matrix (W), initializes 
    the H matrix, and runs the SymNMF optimization via the C extension module. 
    Finally, it assigns each data point to a cluster based on the maximum 
    value in its corresponding row within the final H matrix.

    Args:
        data (numpy.ndarray): The original data points to be clustered.
        k (int): The number of clusters to form.

    Returns:
        list: A list of integers representing the cluster labels (IDs) 
              assigned to each data point.
    """
    W = symnmfmodule.norm(data.tolist())
    H_initial_matrix = user_input.H_initialization(W, k)
    H_final_matrix = symnmfmodule.symnmf(H_initial_matrix, W)

    return [row.index(max(row)) for row in H_final_matrix]


def print_symnmf_score(data, symnmf_labels):
    """
    Calculates and prints the silhouette score for the SymNMF clustering results.

    Args:
        data (list or numpy.ndarray): The original data points.
        symnmf_labels (list): The cluster labels assigned to each data point by SymNMF.
    """
    symnmf_score = silhouette_score(data, symnmf_labels)
    print("nmf: {:.4f}".format(symnmf_score))

if __name__ == "__main__":
    main()