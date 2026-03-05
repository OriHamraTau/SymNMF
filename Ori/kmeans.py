import sys
import math
import copy

Epsilon = 0.0001

def main():
    validate_length_of_input()
    k, iter, vectors = get_arguments()
    validate_arguments(k, iter, vectors)
    centroids = initialize_centroids(k, vectors)
    vectors_per_centroid = intialize_vectors_per_centroid(k)
    centroids = k_means(centroids, vectors, iter, vectors_per_centroid)
    printcentroids(centroids)
    

def validate_length_of_input():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("An Error Has Occurred")
        sys.exit(1)

def get_arguments():
    # read K and iter from command line arguments
    if not sys.argv[1].isdigit():
        print("Incorrect number of clusters!")
        sys.exit(1)
    k = int(sys.argv[1])
    if len(sys.argv) < 3:
        iter = 400
    else:
        if not sys.argv[2].isdigit():
            print("Incorrect maximum iteration!")
            sys.exit(1)
        iter = int(sys.argv[2])

    # read vectors from txt file
    data = sys.stdin.readlines()
    vectors  =  [list(map(float ,line.strip().split(","))) for line in data]

    return k, iter, vectors

def validate_arguments(k, iter, vectors):
    n = len(vectors)
    if k<=1 or k>=n:
        print("Incorrect number of clusters!")
        sys.exit(1)
    if iter<=1 or iter>=800:
        print("Incorrect maximum iteration!")
        sys.exit(1)

    # any others validation - empty txt file, vector dimension mismatch
    if n == 0 or not all(len(vec) == len(vectors[0]) for vec in vectors):
        print("An Error Has Occurred")
        sys.exit(1)


def initialize_centroids(k, vectors):
    # intialize centroids by selecting the first K vectors
    centroids = {i: vectors[i] for i in range(k)}
    return centroids

def intialize_vectors_per_centroid(k):
    vectors_per_centroid = {i: [] for i in range(k)}
    return vectors_per_centroid

def euclidean_distance(vec1, vec2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

def get_centroid_id_per_vector(vector, centroids):
    min_distance = float('inf')
    assigned_centroid = -1

    for centroid_id, centroid in centroids.items():
        distance = euclidean_distance(vector, centroid)
        if distance < min_distance:
            min_distance = distance
            assigned_centroid = centroid_id
    
    return assigned_centroid

def assign_vectors_to_centroids(vectors, centroids, vectors_per_centroid):
    # clear previous assignments
    for centroid_id in vectors_per_centroid:
        vectors_per_centroid[centroid_id] = []
    # assign each vector to the closest centroid   
    for vector in vectors:
        centroid_id = get_centroid_id_per_vector(vector, centroids)
        vectors_per_centroid[centroid_id].append(vector)

def update_centroids_mk(centroids, vectors_per_centroid):
    for centroid_id, vectors in vectors_per_centroid.items():
        new_centroid = [sum(coordinate)/ len(vectors) for coordinate in zip(*vectors)] # calculate mk for each dimension
        centroids[centroid_id] = new_centroid

def convergence(old_centroid, new_centroid):
    for centroid_id in old_centroid:
        if euclidean_distance(old_centroid[centroid_id], new_centroid[centroid_id]) >= Epsilon:
            return False
    return True
    


def k_means(centroids, vectors, iter, vectors_per_centroid):
    curr_iter = 0
    converged = False

    while curr_iter < iter and not converged:
        old_centroids = copy.deepcopy(centroids)
        assign_vectors_to_centroids(vectors, centroids, vectors_per_centroid)
        update_centroids_mk(centroids, vectors_per_centroid)

        converged = convergence(old_centroids, centroids)
        curr_iter += 1
    return centroids

def printcentroids(centroid):
    for i in range(len(centroid)):
        print(','.join("%.4f" % coord for coord in centroid[i]))

if __name__ == "__main__":
    main()