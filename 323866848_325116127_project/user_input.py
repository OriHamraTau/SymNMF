import numpy as np
import math
import sys

goal_set = {"symnmf", "sym", "ddg", "norm"}

def validate_length_of_input(k):
    """
    Validates the number of command-line arguments.

    Args:
        k (int): The expected number of arguments.
    """
    if len(sys.argv) != k:
        print("An Error Has Occurred")
        sys.exit(1) 


def get_arguments(input_length):
    """
    Parses and returns the command-line arguments.

    Args:
        input_length (int): The number of provided arguments.

    Returns:
        tuple: (k, goal, file_name) if input_length is 4, else (k, file_name).
    """
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
    """
    Validates that the number of clusters (k) is within the valid range.

    Args:
        k (int): The requested number of clusters.
        n (int): The total number of data points.
    """
    if k <= 1 or k >= n:
        print("An Error Has Occurred")
        sys.exit(1)


def validate_goal(goal):
    """
    Validates that the requested goal is supported.

    Args:
        goal (str): The requested operation goal.
    """
    if goal not in goal_set:
        print("An Error Has Occurred")
        sys.exit(1)


def validate_file(file_name):
    """
    Validates that the input file has a .txt extension.

    Args:
        file_name (str): The input file path.
    """
    if not (isinstance(file_name, str) and file_name.endswith(".txt")):
        print("An Error Has Occurred")
        sys.exit(1)


def H_initialization(W, k):
    """
    Initializes the H matrix for the SymNMF algorithm.

    Args:
        W (list): The normalized similarity matrix.
        k (int): The number of clusters.

    Returns:
        list: The initialized H matrix as a list of lists.
    """
    n = len(W)
    total = sum(sum(row) for row in W)
    m = total / (n*n)
    np.random.seed(1234)
    initial_H = np.random.uniform(0, 2*math.sqrt(m/k), size=(n,k))
    return initial_H.tolist() # return list of lists
