#from kmeans_pytorch import kmeans
import math
import torch
from time import time
#import numpy as np
from matplotlib import pyplot as plt
import os
import imageio
from torch import nn

class noKMeans:
    '''
    Kmeans clustering algorithm implemented with PyTorch
    Parameters:
      n_clusters: int,
        Number of clusters
      max_iter: int, default: 100
        Maximum number of iterations
      tol: float, default: 0.0001
        Tolerance

      verbose: int, default: 0
        Verbosity
      mode: {'euclidean', 'cosine'}, default: 'euclidean'
        Type of distance measure
      minibatch: {None, int}, default: None
        Batch size of MinibatchKmeans algorithm
        if None perform full KMeans algorithm

    Attributes:
      centroids: torch.Tensor, shape: [n_clusters, n_features]
        cluster centroids
    '''

    def __init__(self, n_clusters, max_iter=100, tol=0.0001, verbose=0, mode="euclidean", minibatch=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.mode = mode
        self.minibatch = minibatch
        self._loop = False
        self._show = False

        try:
            import PYNVML
            self._pynvml_exist = True
        except ModuleNotFoundError:
            self._pynvml_exist = False

        self.centroids = None

    @staticmethod
    def cos_sim(a, b):
        """
          Compute cosine similarity of 2 sets of vectors
          Parameters:
          a: torch.Tensor, shape: [m, n_features]
          b: torch.Tensor, shape: [n, n_features]
        """
        a_norm = a.norm(dim=-1, keepdim=True)
        b_norm = b.norm(dim=-1, keepdim=True)
        a = a / (a_norm + 1e-8)
        b = b / (b_norm + 1e-8)
        return a @ b.transpose(-2, -1)

    @staticmethod
    def euc_sim(a, b):
        """
          Compute euclidean similarity of 2 sets of vectors
          Parameters:
          a: torch.Tensor, shape: [m, n_features]
          b: torch.Tensor, shape: [n, n_features]
        """
        return 2 * a @ b.transpose(-2, -1) - (a ** 2).sum(dim=1)[..., :, None] - (b ** 2).sum(dim=1)[..., None, :]

    def remaining_memory(self):
        """
          Get remaining memory in gpu
        """
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if self._pynvml_exist:
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            remaining = info.free
        else:
            remaining = torch.cuda.memory_allocated()
        return remaining

    def max_sim(self, a, b):
        """
          Compute maximum similarity (or minimum distance) of each vector
          in a with all of the vectors in b
          Parameters:
          a: torch.Tensor, shape: [m, n_features]
          b: torch.Tensor, shape: [n, n_features]
        """
        device = a.device.type
        batch_size = a.shape[0]
        if self.mode == 'cosine':
            sim_func = self.cos_sim
        elif self.mode == 'euclidean':
            sim_func = self.euc_sim

        if device == 'cpu':
            sim = sim_func(a, b)
            max_sim_v, max_sim_i = sim.max(dim=-1)
            return max_sim_v, max_sim_i
        else:
            if a.dtype == torch.float:
                expected = a.shape[0] * a.shape[1] * b.shape[0] * 4
            elif a.dtype == torch.half:
                expected = a.shape[0] * a.shape[1] * b.shape[0] * 2
            ratio = math.ceil(expected / self.remaining_memory())
            subbatch_size = math.ceil(batch_size / ratio)
            msv, msi = [], []
            for i in range(ratio):
                if i * subbatch_size >= batch_size:
                    continue
                sub_x = a[i * subbatch_size: (i + 1) * subbatch_size]
                sub_sim = sim_func(sub_x, b)
                sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
                del sub_sim
                msv.append(sub_max_sim_v)
                msi.append(sub_max_sim_i)
            if ratio == 1:
                max_sim_v, max_sim_i = msv[0], msi[0]
            else:
                max_sim_v = torch.cat(msv, dim=0)
                max_sim_i = torch.cat(msi, dim=0)
            return max_sim_v, max_sim_i

    def fit_predict(self, X, centroids=None):
        """
          Combination of fit() and predict() methods.
          This is faster than calling fit() and predict() seperately.
          Parameters:
          X: torch.Tensor, shape: [n_samples, n_features]
          centroids: {torch.Tensor, None}, default: None
            if given, centroids will be initialized with given tensor
            if None, centroids will be randomly chosen from X
          Return:
          labels: torch.Tensor, shape: [n_samples]
        """
        batch_size, emb_dim = X.shape
        device = X.device.type
        start_time = time()
        if centroids is None:
            self.centroids = X[np.random.choice(batch_size, size=[self.n_clusters], replace=False)]
        else:
            self.centroids = centroids
        num_points_in_clusters = torch.ones(self.n_clusters, device=device)
        closest = None
        for i in range(self.max_iter):
            iter_time = time()
            if self.minibatch is not None:
                x = X[np.random.choice(batch_size, size=[self.minibatch], replace=False)]
            else:
                x = X
            closest = self.max_sim(a=x, b=self.centroids)[1]
            matched_clusters, counts = closest.unique(return_counts=True)

            c_grad = torch.zeros_like(self.centroids)
            if self._loop:
                for j, count in zip(matched_clusters, counts):
                    c_grad[j] = x[closest == j].sum(dim=0) / count
            else:
                if self.minibatch is None:
                    expanded_closest = closest[None].expand(self.n_clusters, -1)
                    mask = (expanded_closest == torch.arange(self.n_clusters, device=device)[:, None]).float()
                    c_grad = mask @ x / mask.sum(-1)[..., :, None]
                    c_grad[c_grad != c_grad] = 0  # remove NaNs
                else:
                    expanded_closest = closest[None].expand(len(matched_clusters), -1)
                    mask = (expanded_closest == matched_clusters[:, None]).float()
                    c_grad[matched_clusters] = mask @ x / mask.sum(-1)[..., :, None]

                # if x.dtype == torch.float:
                #   expected = closest.numel() * len(matched_clusters) * 5 # bool+float
                # elif x.dtype == torch.half:
                #   expected = closest.numel() * len(matched_clusters) * 3 # bool+half
                # if device == 'cpu':
                #   ratio = 1
                # else:
                #   ratio = math.ceil(expected / self.remaining_memory() )
                # # ratio = 1
                # subbatch_size = math.ceil(len(matched_clusters)/ratio)
                # for j in range(ratio):
                #   if j*subbatch_size >= batch_size:
                #     continue
                #   sub_matched_clusters = matched_clusters[j*subbatch_size: (j+1)*subbatch_size]
                #   sub_expanded_closest = closest[None].expand(len(sub_matched_clusters), -1)
                #   sub_mask = (sub_expanded_closest==sub_matched_clusters[:, None]).to(x.dtype)
                #   sub_prod = sub_mask @ x / sub_mask.sum(1)[:, None]
                #   c_grad[sub_matched_clusters] = sub_prod
            error = (c_grad - self.centroids).pow(2).sum()
            if self.minibatch is not None:
                lr = 1 / num_points_in_clusters[:, None] * 0.9 + 0.1
                # lr = 1/num_points_in_clusters[:,None]**0.1
            else:
                lr = 1
            num_points_in_clusters[matched_clusters] += counts
            self.centroids = self.centroids * (1 - lr) + c_grad * lr
            if self.verbose >= 2:
                print('iter:', i, 'error:', error.item(), 'time spent:', round(time() - iter_time, 4))
            if error <= self.tol:
                break

        # SCATTER
        #f self._show:
        #    if self.mode is "cosine":
        #        sim = self.cos_sim(x, self.centroids)
        #    elif self.mode is "euclidean":
        #        sim = self.euc_sim(x, self.centroids)
        #    closest = sim.argmax(dim=-1)
        #    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=closest.cpu(), marker='.', cmap='hsv')
        #    plt.scatter(self.centroids[:,0].cpu(), self.centroids[:,1].cpu(), marker='o', cmap='red')
        #    plt.show()
        # END SCATTER
        if self.verbose >= 1:
            print(
                f'used {i + 1} iterations ({round(time() - start_time, 4)}s) to cluster {batch_size} items into {self.n_clusters} clusters')
        return closest, self.centroids

    def predict(self, X):
        """
          Predict the closest cluster each sample in X belongs to
          Parameters:
          X: torch.Tensor, shape: [n_samples, n_features]
          Return:
          labels: torch.Tensor, shape: [n_samples]
        """
        return self.max_sim(a=X, b=self.centroids)[1]

    def fit(self, X, centroids=None):
        """
          Perform kmeans clustering
          Parameters:
          X: torch.Tensor, shape: [n_samples, n_features]
        """
        self.fit_predict(X, centroids)


class KMeansPP(nn.Module):
    def __init__(self, n_clusters, max_iter=100, tol=0.0001, return_lbl=False, device=torch.device('cuda')):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.return_lbl = return_lbl
        self.centroids = None
        self.lbl = None
        self.device = device

    def forward(self, X, centroids=None):
        self.centroids = self.centroids_init(X, centroids)
        for i in range(self.max_iter):
            centroid_added = False
            new_centroids, used_centroids = self.kmeans_step(X, self.centroids)
            centr_shift = self.calc_centr_shift(new_centroids, used_centroids)
            #new_centroids = new_centroids[0: -3, :]
            #print(new_centroids.shape)
            if new_centroids.shape[0] < self.n_clusters:
                self.centroids = self.centroids_init(X, new_centroids)
                centroid_added = True
                #print("centroid_Added", centroid_added)
            else:
                self.centroids = new_centroids
            if (centr_shift <= self.tol) and (not centroid_added):
                #print("Iterations taken:", i)
                if self.return_lbl:
                    _, lbl = self.calc_dist_lbl(X, self.centroids)
                    return self.centroids, lbl
                return self.centroids
        #print("Iterations taken:", i)
        if self.return_lbl:
            _, lbl = self.calc_dist_lbl(X, self.centroids)
            return self.centroids, lbl
        return self.centroids

    def kmeans_step(self, X, centroids):
        old_centroids = centroids
        _, lbl = self.calc_dist_lbl(X, old_centroids)
        lbl_mask, elem_per_lbl, used_lbls = self.create_lblmask_elemperlbl_usedlbl(lbl)
        x_rep = X.repeat(self.n_clusters, 1, 1)
        einsum = torch.einsum('abc,ab->abc', x_rep, lbl_mask)
        lbl_einsum_sum = torch.sum(einsum, dim=1)
        mean_sum = torch.divide(lbl_einsum_sum, elem_per_lbl)
        new_centroids = mean_sum[[~torch.any(mean_sum.isnan(), dim=1)]]
        used_centroids = old_centroids[[~torch.any(mean_sum.isnan(), dim=1)]]
        return new_centroids, used_centroids,

    def centroids_init(self, X, centroids):
        if centroids is None:
            centroids = X[torch.randint(0, X.shape[0], (1,))]
        while centroids.shape[0] < self.n_clusters:
            outlier_coor = self.calc_outlier_coor(X, centroids)
            outlier = X[outlier_coor, :][None, ...]
            centroids = torch.cat((centroids, outlier), dim=0)
        return centroids

    def calc_dist_lbl(self, X, centroids):
        sq_dist = torch.cdist(centroids, X, 2)
        min_sq_dist, lbl = torch.min(sq_dist, dim=0)
        return min_sq_dist, lbl

    def calc_outlier_coor(self, X, centroids):
        sq_dist, _ = self.calc_dist_lbl(X, centroids)
        argmax_dist = torch.argmax(sq_dist)
        return argmax_dist

    def create_lblmask_elemperlbl_usedlbl(self, lbl):
        used_lbls = torch.arange(self.n_clusters, device=self.device).view(self.n_clusters, 1)
        lbl_mask = used_lbls.repeat(1, lbl.shape[0])
        lbl_mask = torch.subtract(lbl_mask, lbl)
        lbl_mask = lbl_mask.eq(0)#.type(torch.int)
        elem_per_lbl = torch.sum(lbl_mask, dim=1).view(self.n_clusters, 1)
        return lbl_mask, elem_per_lbl, used_lbls

    def calc_centr_shift(self, centroids_1, centroids_2):
        shift = torch.subtract(centroids_1, centroids_2).abs().pow(2)
        shift = torch.sum(shift)
        return shift

relevant_path = "/home/bernard/PycharmProjects/uORF/uORF-main/room_chair_train/"
included_extensions = ['png']
file_names = [fn for fn in os.listdir(relevant_path)
              if any(fn.endswith(ext) for ext in included_extensions)]
cuda = torch.device('cuda:0')
kmeans = KMeansPP(10, 500, 0.0001, False, cuda)
for fi in zip(file_names, range(len(file_names))):
    #print(f)
    f = fi[0]
    if fi[1]%200 == 0:
        print(fi[1])
    im = imageio.imread(relevant_path + f)
    #plt.imshow(im[:, :, :3])
    #plt.show()
    im = torch.from_numpy(im[:64, :64, :3]).flatten(start_dim=0, end_dim=1).type(torch.FloatTensor).cuda()
    #im = im - (torch.min(im) + 4)
    #print(torch.min(im))
    #centroids_init = torch.rand((50, 3))
    #centroids_init = centroids_init + (im.max() + im.min())//2
    centroids = kmeans(im, None).cpu()
    im = im.cpu()
    # def forward(self, X, centroids=None):
    #print(centroids)
    #print(labels)
    if fi[1] % 200 == 0:
        x = im[:, 0]
        y = im[:, 1]
        z = im[:, 2]
        xc = centroids[:, 0]
        yc = centroids[:, 1]
        zc = centroids[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, marker="x", c="red")
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xc, yc, zc, c="black")
        plt.show()

"""
relevant_path = "/home/bernard/PycharmProjects/uORF/uORF-main/room_chair_train/"
included_extensions = ['png']
file_names = [fn for fn in os.listdir(relevant_path)
              if any(fn.endswith(ext) for ext in included_extensions)]
file_names = ["01791_sc0447_az03.png"]
#kmeans = KMeans(n_clusters=10, max_iter=100, mode='euclidean', verbose=1)

for f in file_names:
    im = imageio.imread(relevant_path + f)
    print(f)
    im = torch.from_numpy(im[:, :, :2]).flatten(start_dim=0, end_dim=1).type(torch.FloatTensor)
    cluster_ids_x, cluster_centers = kmeans(
        X=im, num_clusters=10, distance='euclidean', device=torch.device('cuda:0')
    )
    x = im[:, 0]
    y = im[:, 1]
    #z = im[:, 2]
    xc = cluster_centers[:, 0]
    yc = cluster_centers[:, 1]
    #zc = cluster_centers[:, 0]
    plt.scatter(x, y, marker="x", c="red")
    plt.scatter(xc, yc, c="black")
    plt.show()
"""

