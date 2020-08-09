import numpy as np


import sys
print(sys.getrecursionlimit())

def basicDBScan(Data, EPS, RNG, MinPts, DistFunc = None):
    """
    Parameters:
    ----------
    Data: array-like
        array of points of the shape (num_samples, num_dimensions)
    Eps: float
        distance around each point to be considered. Passed to distance function if not default.
    MinPts: int
        minimum number of points required for a point to be added to a cluster.
    DistFunc: function, default = None
        pass if want non-default distance calculating function
    """
    
    cluster_ids = np.zeros(len(Data))
    indices = np.arange(len(Data))
    
    def radial_dist(Datapt, Data, EPS):
           return (np.sum((Data - Datapt)**2, axis = 1)**.5 <= EPS)
    
    if DistFunc == None:
        DistFunc = radial_dist
    
    def pick_Random(indices, Data, RNG, cluster_ids):
        index = RNG.choice(indices[cluster_ids == False])
        point = Data[index]
        
        return (index,point)
    
    def recursive_Cluster(pt):
            distance_bools = DistFunc(pt, Data, EPS)
            if np.count_nonzero(distance_bools) >= MinPts:
                cluster_ids[indices[distance_bools]] = cluster_count
                pts = Data[indices[distance_bools]]
                for pt in pts:
                    recursive_Cluster(pt)
    
    count = 0
    cluster_count = 0
    while np.any(cluster_ids == 0):
        count += 1
        init_index, init_point = pick_Random(indices, Data, RNG, cluster_ids)
        if np.count_nonzero(DistFunc(init_point, Data, EPS)) >= MinPts:
            print('cluster found')
            cluster_count += 1
            recursive_Cluster(init_point)
        else:
            cluster_ids[init_index] = -1
    
    return cluster_ids

# P. Fränti and S. Sieranoja A3 Dataset of 50 clusters
import matplotlib.pyplot as plt

a3 = np.loadtxt(r'./datasets/Clustering - a3 (k =50).txt')

print('shape: ',a3.shape)
print('Min and max x-values: ', np.min(a3[:, 0]), np.max(a3[:, 0]))
print('Min and max y-values: ', np.min(a3[:, 1]), np.max(a3[:, 1]))

plt.scatter(a3[:, 0], a3[:, 1])
plt.title('A3 Dataset from P. Fränti and S. Sieranoja')
plt.show()

a3_norm = np.array(a3, copy = True)
a3_norm[:, 0] /= np.max(a3[:, 0])
a3_norm[:, 1] /= np.max(a3[:, 0])

plt.scatter(a3_norm[:, 0], a3_norm[:, 1])
plt.title("A3 Dataset Normalized so that Max(x) is 1")
plt.show()

RNG = np.random.default_rng()

arr = basicDBScan(a3_norm, EPS = .05, RNG = RNG, MinPts = 200, DistFunc = None)

print(max(arr))