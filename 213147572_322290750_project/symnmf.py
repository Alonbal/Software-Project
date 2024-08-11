
import sys
import symnmfmodule
import numpy as np
from math import sqrt

ERR_MSG = "An Error Has Occurred"
C_GOALS = ["sym", "ddg", "norm"]

MAX_ITER = 300
EPS = 1e-4
BETA = 0.5

def main():
    if len(sys.argv) != 4 or not sys.argv[1].isnumeric:
        print(ERR_MSG)
        return 

    k = int(sys.argv[1])
    goal = sys.argv[2]
    file_name = sys.argv[3]

    if goal in C_GOALS:
        result = symnmfmodule.calc_mat(file_name, goal)
    
    elif goal == "symnmf":
        norm = symnmfmodule.calc_mat(file_name, "norm")
        result = optimize(norm, k)

    else:
        print(ERR_MSG)
        return

    pretty_print(result)


def pretty_print(matrix):
    print("\n".join(", ".join(f"{item:.4f}" for item in row) for row in matrix))

def optimize(W, k):
    W = np.array(W)
    np.random.seed(0)
    iter = 0

    m = W.mean()
    n = W.shape[0]

    H = np.random.uniform(0.0, 2 * sqrt(m/k), (n, k))
    Hnext = H

    while (((H - Hnext)**2).sum() >= EPS or iter < MAX_ITER or iter == 0): 
        H = Hnext
        iter += 1
        WH = np.matmul(W, H)
        HHH = np.matmul(H, np.matmul(H.T, H))
        factor = (WH / HHH) * BETA + (1 - BETA)
        Hnext = factor * H

    return H

if __name__ == "__main__":
    main()
