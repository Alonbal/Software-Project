import pandas as pd
import numpy as np
import sys
import mykmeanssp as mk


def load_data(file_name_1, file_name_2):
    try:
        data1 = pd.read_csv(file_name_1, header=None)
        data2 = pd.read_csv(file_name_2, header=None)
    except FileNotFoundError:
        return None

    data = pd.merge(data1, data2, on=0)

    data = data.sort_values(by=0, ascending=True)

    return (data.values)


def kmeans_init_centers(data, K):
    np.random.seed(0)
    unind_data = data[:, 1:]
    index = np.random.choice(data.shape[0])
    center_indexes = [index]

    for _ in range(K-1):
        distances = np.array([float("inf") for _ in range(data.shape[0])])
        for center_ind in center_indexes:
            difference = unind_data - unind_data[center_ind]

            # for each point, calculate its distance from the current center
            dist_from_center = np.sum(difference ** 2, axis = 1) ** (1/2)

            # apply minimum row by row
            distances = np.minimum(distances, dist_from_center)  

        sum_dists = np.sum(distances) 
        probabilities = distances / sum_dists

        index = np.random.choice(data.shape[0], p=probabilities)

        center_indexes.append(index)

    return data[center_indexes]


def main():
    
    vars = sys.argv[1:]
    mx_iter = 300

    clust_err_msg = "Invalid number of clusters!"
    gen_err_msg = "An Error Has Occurred"
    iter_err_msg = "Invalid maximum iteration!"

    if len(vars) > 5 or len(vars) < 4:
        print(gen_err_msg)
        return 1

    if vars[0].isnumeric():
        K = int(vars[0])
    else:
        print(clust_err_msg)
        return 1

    eps = float(vars[-3])
    file_name_1 = vars[-2]
    file_name_2 = vars[-1]

    data = load_data(file_name_1, file_name_2)
    if data is None:
        print(gen_err_msg)
        return 1
    
    if K >= len(data):
        print(clust_err_msg)
        return 1

    if len(vars) == 5:
        if vars[1].isnumeric():
            mx_iter = int(vars[1])
        else:
            print(iter_err_msg)
            return 1
        
    if mx_iter >= 1000 or mx_iter <= 1:
        print(iter_err_msg)
        return 1
    
    init_centers = kmeans_init_centers(data, K)

    print(",".join(map(lambda x: str(int(x)), init_centers[:, 0].tolist())))

    init_centers = init_centers[:, 1:].tolist()
    data = data[:, 1:].tolist()
    try:
        results = mk.fit(init_centers, data, mx_iter, eps)
    except Exception:
        print(gen_err_msg)
        return 1

    for point in results:
        print(",".join("{:.4f}".format(x) for x in point))

    return 0

if __name__ == "__main__":
    main()