import sys
import math
import random
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics.cluster import contingency_matrix
from matplotlib import pyplot as plt


class TopClustering:
    """Topological clustering.
    
    Attributes:
        n_clusters: 
          The number of clusters.
        top_relative_weight:
          Relative weight between the geometric and topological terms.
          A floating point number between 0 and 1.
        max_iter_alt:
          Maximum number of iterations for the topological clustering.
        max_iter_interp:
          Maximum number of iterations for the topological interpolation.
        learning_rate:
          Learning rate for the topological interpolation.
        
    Reference:
        Songdechakraiwut, Tananun, Bryan M. Krause, Matthew I. Banks, Kirill V. Nourski, and Barry D. Van Veen. 
        "Fast topological clustering with Wasserstein distance." 
        International Conference on Learning Representations (ICLR). 2022.
    """

    def __init__(self, n_clusters, top_relative_weight, max_iter_alt,
                 max_iter_interp, learning_rate):
        self.n_clusters = n_clusters
        self.top_relative_weight = top_relative_weight
        self.max_iter_alt = max_iter_alt
        self.max_iter_interp = max_iter_interp
        self.learning_rate = learning_rate

    def fit_predict(self, data):
        """Computes topological clustering and predicts cluster index for each sample.
        
            Args:
                data:
                  Training instances to cluster.
                  
            Returns:
                Cluster index each sample belongs to.
        """
        data = np.asarray(data)
        n_node = data.shape[1]
        n_edges = math.factorial(n_node) // math.factorial(2) // math.factorial(
            n_node - 2)  # n_edges = (n_node choose 2)
        n_births = n_node - 1
        self.weight_array = np.append(
            np.repeat(1 - self.top_relative_weight, n_edges),
            np.repeat(self.top_relative_weight, n_edges))

        # Networks represented as vectors concatenating geometric and topological info
        X = []
        for adj in data:
            X.append(self._vectorize_geo_top_info(adj))
        X = np.asarray(X)

        # Random initial condition
        self.centroids = X[random.sample(range(X.shape[0]), self.n_clusters)]

        # Assign the nearest centroid index to each data point
        assigned_centroids = self._get_nearest_centroid(
            X[:, None, :], self.centroids[None, :, :])
        prev_assigned_centroids = assigned_centroids

        for it in range(self.max_iter_alt):
            for cluster in range(self.n_clusters):
                # Previous iteration centroid
                prev_centroid = np.zeros((n_node, n_node))
                prev_centroid[np.triu_indices(
                    prev_centroid.shape[0],
                    k=1)] = self.centroids[cluster][:n_edges]

                # Determine data points belonging to each cluster
                cluster_members = X[assigned_centroids == cluster]

                # Compute the sample mean and top. centroid of the cluster
                cc = cluster_members.mean(axis=0)
                sample_mean = np.zeros((n_node, n_node))
                sample_mean[np.triu_indices(sample_mean.shape[0],
                                            k=1)] = cc[:n_edges]
                top_centroid = cc[n_edges:]
                top_centroid_birth_set = top_centroid[:n_births]
                top_centroid_death_set = top_centroid[n_births:]

                # Update the centroid
                try:
                    cluster_centroid = self._top_interpolation(
                        prev_centroid, sample_mean, top_centroid_birth_set,
                        top_centroid_death_set)
                    self.centroids[cluster] = self._vectorize_geo_top_info(
                        cluster_centroid)
                except:
                    print(
                        'Error: Possibly due to the learning rate is not within appropriate range.'
                    )
                    sys.exit(1)

            # Update the cluster membership
            assigned_centroids = self._get_nearest_centroid(
                X[:, None, :], self.centroids[None, :, :])

            # Compute and print loss as it is progressively decreasing
            loss = self._compute_top_dist(
                X, self.centroids[assigned_centroids]).sum() / len(X)
            print('Iteration: %d -> Loss: %f' % (it, loss))

            if (prev_assigned_centroids == assigned_centroids).all():
                break
            else:
                prev_assigned_centroids = assigned_centroids
        return assigned_centroids

    def _vectorize_geo_top_info(self, adj):
        birth_set, death_set = self._compute_birth_death_sets(
            adj)  # topological info
        vec = adj[np.triu_indices(adj.shape[0], k=1)]  # geometric info
        return np.concatenate((vec, birth_set, death_set), axis=0)

    def _compute_birth_death_sets(self, adj):
        """Computes birth and death sets of a network."""
        mst, nonmst = self._bd_demomposition(adj)
        birth_ind = np.nonzero(mst)
        death_ind = np.nonzero(nonmst)
        return np.sort(mst[birth_ind]), np.sort(nonmst[death_ind])

    def _bd_demomposition(self, adj):
        """Birth-death decomposition."""
        eps = np.nextafter(0, 1)
        adj[adj == 0] = eps
        adj = np.triu(adj, k=1)
        Xcsr = csr_matrix(-adj)
        Tcsr = minimum_spanning_tree(Xcsr)
        mst = -Tcsr.toarray()  # reverse the negative sign
        nonmst = adj - mst
        return mst, nonmst

    def _get_nearest_centroid(self, X, centroids):
        """Determines cluster membership of data points."""
        dist = self._compute_top_dist(X, centroids)
        nearest_centroid_index = np.argmin(dist, axis=1)
        return nearest_centroid_index

    def _compute_top_dist(self, X, centroid):
        """Computes the pairwise top. distances between networks and centroids."""
        return np.dot((X - centroid)**2, self.weight_array)

    def _top_interpolation(self, init_centroid, sample_mean,
                           top_centroid_birth_set, top_centroid_death_set):
        """Topological interpolation."""
        curr = init_centroid
        for _ in range(self.max_iter_interp):
            # Geometric term gradient
            geo_gradient = 2 * (curr - sample_mean)

            # Topological term gradient
            sorted_birth_ind, sorted_death_ind = self._compute_optimal_matching(
                curr)
            top_gradient = np.zeros_like(curr)
            top_gradient[sorted_birth_ind] = top_centroid_birth_set
            top_gradient[sorted_death_ind] = top_centroid_death_set
            top_gradient = 2 * (curr - top_gradient)

            # Gradient update
            curr -= self.learning_rate * (
                (1 - self.top_relative_weight) * geo_gradient +
                self.top_relative_weight * top_gradient)
        return curr

    def _compute_optimal_matching(self, adj):
        mst, nonmst = self._bd_demomposition(adj)
        birth_ind = np.nonzero(mst)
        death_ind = np.nonzero(nonmst)
        sorted_temp_ind = np.argsort(mst[birth_ind])
        sorted_birth_ind = tuple(np.array(birth_ind)[:, sorted_temp_ind])
        sorted_temp_ind = np.argsort(nonmst[death_ind])
        sorted_death_ind = tuple(np.array(death_ind)[:, sorted_temp_ind])
        return sorted_birth_ind, sorted_death_ind


#############################################
################### Demo ####################
#############################################
def random_modular_graph(d, c, p, mu, sigma):
    """Simulated modular network.
    
        Args:
            d: Number of nodes.
            c: Number of clusters/modules.
            p: Probability of attachment within module.
            mu, sigma: Used for random edge weights.
            
        Returns:
            Adjacency matrix.
    """
    adj = np.zeros((d, d))  # adjacency matrix
    for i in range(1, d + 1):
        for j in range(i + 1, d + 1):
            module_i = math.ceil(c * i / d)
            module_j = math.ceil(c * j / d)

            # Within module
            if module_i == module_j:
                if random.uniform(0, 1) <= p:
                    w = np.random.normal(mu, sigma)
                    adj[i - 1][j - 1] = max(w, 0)
                else:
                    w = np.random.normal(0, sigma)
                    adj[i - 1][j - 1] = max(w, 0)

            # Between modules
            else:
                if random.uniform(0, 1) <= 1 - p:
                    w = np.random.normal(mu, sigma)
                    adj[i - 1][j - 1] = max(w, 0)
                else:
                    w = np.random.normal(0, sigma)
                    adj[i - 1][j - 1] = max(w, 0)
    return adj


def purity_score(labels_true, labels_pred):
    mtx = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(mtx, axis=0)) / np.sum(mtx)


def main():
    np.random.seed(0)
    random.seed(0)

    # Generate a dataset comprising simulated modular networks
    dataset = []
    labels_true = []
    n_network = 20
    n_node = 60
    p = 0.7
    mu = 1
    sigma = 0.5
    for module in [2, 3, 5]:
        for _ in range(n_network):
            adj = random_modular_graph(n_node, module, p, mu, sigma)
            # Uncomment lines below for visualization
            # plt.imshow(adj, vmin=0, vmax=2, cmap='YlOrRd')
            # plt.colorbar()
            # plt.show()
            dataset.append(adj)
            labels_true.append(module)

    # Topological clustering
    n_clusters = 3
    top_relative_weight = 0.99  # 'top_relative_weight' between 0 and 1
    max_iter_alt = 300
    max_iter_interp = 300
    learning_rate = 0.05
    print('Topological clustering\n----------------------')
    labels_pred = TopClustering(n_clusters, top_relative_weight, max_iter_alt,
                                max_iter_interp,
                                learning_rate).fit_predict(dataset)
    print('\nResults\n-------')
    print('True labels:', np.asarray(labels_true))
    print('Pred indices:', labels_pred)
    print('Purity score:', purity_score(labels_true, labels_pred))


if __name__ == '__main__':
    main()
