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
    if goal == "symnmf":
        W = symnmfmodule.norm(vectors)
        H = user_input.H_intialization(W, k)
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
    for row in mat:
        print(",".join(["{:.4f}".format(val) for val in row]))


if __name__ == "__main__":
    main()