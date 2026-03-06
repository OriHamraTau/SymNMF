import numpy as np
import math
import sys

goal_set = {"symnmf", "sym", "ddg", "norm"}

def validate_length_of_input(k):
    if len(sys.argv) != k:
        print("An Error Has Occurred")
        sys.exit(1) 

def get_arguments(input_length):
    if not sys.argv[1].isdigit():
        print("An Error Has Occurred")
        sys.exit(1)
    k = int(sys.argv[1])
    if input_length == 4:
        goal = sys.argv[2]
        file_name = sys.argv[3]
        return k, goal , file_name
    else:
        file_name = sys.argv[2]
    return k, file_name

def validate_k(k, n):
    if k <= 1 or k >= n:
        print("An Error Has Occurred")
        sys.exit(1)

def validate_goal(goal):
    if goal not in goal_set:
        print("An Error Has Occurred")
        sys.exit(1)

def validate_file(file_name):
    if not (isinstance(file_name, str) and file_name.endswith(".txt")):
        print("An Error Has Occurred")
        sys.exit(1)

def H_intialization(W, k):
    n = len(W)
    total = sum(sum(row) for row in W)
    m = total / (n*n)
    np.random.seed(1234)
    initial_H = np.random.uniform(0, 2*math.sqrt(m/k), size=(n,k))
    return initial_H.tolist() # return list of lists 
