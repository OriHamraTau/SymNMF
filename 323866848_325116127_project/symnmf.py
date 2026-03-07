import sys
import numpy as np
import symnmfmodule
import user_input

Input_Length = 4

def main():
    user_input.validate_length_of_input(Input_Length)
    k, goal, file_name = user_input.get_arguments(Input_Length)

    # load files into NumPy arrays
    data = np.loadtxt(file_name, delimiter=',', ndmin=2)
    # get dimensions
    n, d = data.shape

    user_input.validate_k(k, n)
    user_input.validate_goal(goal)
    user_input.validate_file(file_name)

    required_matrix(data.tolist(), goal, k)

def required_matrix(vectors, goal, k):
    """
    Routes the execution to the appropriate SymNMF C-extension function based on the goal.
    Calculates the required matrix and prints it.

    Args:
        vectors (list): The input data points as a list of lists.
        goal (str): The specific matrix to compute ('symnmf', 'sym', 'ddg', or 'norm').
        k (int): The number of clusters (used only for the 'symnmf' goal).
    """
    if goal == "symnmf":
        W = symnmfmodule.norm(vectors)
        H = user_input.H_inisettialization(W, k)
        result_mat = symnmfmodule.symnmf(H, W)
    elif goal == "sym":
        result_mat = symnmfmodule.sym(vectors)
    elif goal == "ddg":
        result_mat = symnmfmodule.ddg(vectors)
    elif goal == "norm":
        result_mat = symnmfmodule.norm(vectors)
    else:
        print("An Error Has Occurred")
        sys.exit(1)

    print_matrix(result_mat)

def print_matrix(mat):
    """
    Prints a 2D matrix as comma-separated values, formatted to 4 decimal places.

    Args:
        mat (list of lists): The 2D matrix to print.
    """
    for row in mat:
        print(",".join(["{:.4f}".format(val) for val in row]))

if __name__ == "__main__":
    main()