import sys
import math
import copy

EPSILON = 0.001


def initialize_centroids(k, vectors):
    return [copy.deepcopy(vectors[i]) for i in range(k)]

def intialize_vectors_per_centroid(k):
    return [[] for _ in range(k)]

def get_centroid_id_per_vector(vector, centroids):
    min_distance = float("inf")
    closest_centroid = 0
    d = len(vector)
    
    for i, centroid in enumerate(centroids):
        distance = 0
        for j in range(d):
            distance += (vector[j] - centroid[j])**2
        
        distance = math.sqrt(distance)
        if distance < min_distance:
            min_distance = distance
            closest_centroid = i
    return closest_centroid


def k_means(centroids, vectors, iter_limit, vectors_per_centroid=None):
    n = len(vectors)
    k = len(centroids)
    d = len(vectors[0])

    for iter_cnt in range(iter_limit):
        centroids_size = [0 for _ in range(k)]
        centroids_sum = [[0 for _ in range(d)] for _ in range(k)]

        for i in range(n):
            closest_centroid = get_centroid_id_per_vector(vectors[i], centroids)
            
            centroids_size[closest_centroid] += 1
            for j in range(d):
                centroids_sum[closest_centroid][j] += vectors[i][j]
        
        prev_centroids = copy.deepcopy(centroids)
        for i in range(k):
            if centroids_size[i] != 0:
                for j in range(d):
                    centroids[i][j] = centroids_sum[i][j] / centroids_size[i]
        
        is_converged = True
        for i in range(k):
            shifted = 0
            for j in range(d):
                shifted += (prev_centroids[i][j] - centroids[i][j]) ** 2
            if math.sqrt(shifted) >= EPSILON:
                is_converged = False
                break
        
        if is_converged:
            break
            
    return centroids


def main():
    data = sys.stdin.readlines()
    vectors = []
    for line in data:
        if not line.strip(): continue
        try:
            vectors.append([float(x) for x in line.strip().split(",")])
        except ValueError:
            print("An Error Has Occurred")
            sys.exit(1)

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        if k <= 1 or k >= len(vectors):
            print("Incorrect number of clusters!")
            sys.exit(1)
    except:
        print("Incorrect number of clusters!")
        sys.exit(1)

    num_iter = 400
    if len(sys.argv) == 3:
        try:
            num_iter = int(sys.argv[2])
            if num_iter <= 1 or num_iter >= 800:
                print("Incorrect maximum iteration!")
                sys.exit(1)
        except:
            print("Incorrect maximum iteration!")
            sys.exit(1)

    initial_centroids = initialize_centroids(k, vectors)
    final_centroids = k_means(initial_centroids, vectors, num_iter)

    for c in final_centroids:
        print(",".join([f"{x:.4f}" for x in c]))

if __name__ == "__main__":
    main()