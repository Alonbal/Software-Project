
import sys
import math
import symnmf
import symnmfmodule
from sklearn.metrics import silhouette_score

EPS = 0.001

def dist(p1, p2):
    return math.sqrt(sum((a - b)**2 for (a, b) in zip(p1, p2)))

def closest_ind(p, centers):
    return min(range(len(centers)), key=lambda i: dist(p, centers[i]))

def add_to(p1, p2):
    for i in range(len(p1)):
        p1[i] += p2[i]

def divide_by(p1, num):
    for i in range(len(p1)):
        p1[i] /= num

def kmeans(data, K):
    
    mx_iter = 200

    if len(data) == 0:
        print("An Error Has Occured")
        return 

    D = len(data[0])

    if K <= 1 or K >= len(data):
        print("Invalid number of clusters!")
        return 1

    if mx_iter <= 1 or mx_iter >= 1000:
        print("Invalid maximum iteration!")
        return 1

    cluster_points = [data[i][:] for i in range(K)]

    max_diff = EPS + 1
    while mx_iter > 0 and max_diff > EPS:
        new_cluster_points = [[0 for j in range(D)] for i in range(K)]
        bucket_sizes = [0 for i in range(K)]

        for point in data:
            ind = closest_ind(point, cluster_points)
            add_to(new_cluster_points[ind], point)
            bucket_sizes[ind] += 1
        
        for i in range(K):
            divide_by(new_cluster_points[i], bucket_sizes[i]) 
            max_diff = max(max_diff, dist(new_cluster_points[i], cluster_points[i]))
            cluster_points[i] = new_cluster_points[i]

        mx_iter -= 1

    clusters = [0 for _ in range(len(data))]

    for (i, point) in enumerate(data):
        j = closest_ind(point, cluster_points)
        clusters[i] = j

    return clusters

def nmf(file_name, K, data):
    norm = symnmfmodule.calc_mat(file_name, "norm")
    result = symnmf.optimize(norm, K) 

    clusters = [0 for _ in range(len(data))]

    for (i, point) in enumerate(data):
        j = max(range(K), key=lambda j: result[i][j])  
        clusters[i] = j
    
    return clusters

def main(): 

    if len(sys.argv) != 3 or not sys.argv[1].isnumeric:
        print(symnmf.ERR_MSG)
        return

    file_name = sys.argv[2]
    k = int(sys.argv[1])

    try:
        with open(file_name) as file:
            lines = file.readlines()
            data = [[float(x) for x in line.split(",")] for line in lines]
            
    except (OSError, ValueError):
        print(symnmf.ERR_MSG)
        return 1
    
    kmeans_clusters = kmeans(data, k)
    nmf_clusters = nmf(file_name, k, data)

    if kmeans_clusters == None or nmf_clusters == None:
        return None

    kmeans_silhouette_score = silhouette_score(data, kmeans_clusters)
    nmf_silhouette_score = silhouette_score(data, nmf_clusters)

    if kmeans_silhouette_score == None or nmf_silhouette_score == None:
        print(symnmf.ERR_MSG)
        return

    print(f"nmf: {nmf_silhouette_score:.4f}")
    print(f"kmeans: {kmeans_silhouette_score:.4f}")

if __name__ == "__main__":
    main()