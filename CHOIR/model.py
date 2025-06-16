import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import silhouette_score
import scipy.stats
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier

from collections import defaultdict

class CHOIR:

    def __init__(self, adata):
        """
        Initialize the CHOIR model with an AnnData object.
        
        Parameters:
        adata (AnnData): The annotated data matrix.
        """
        self.adata = adata

    @staticmethod
    def optimal_clustering(adata):
        # We'll be using a copy of the original AnnData object.
        # Return the best resolution and labels.
        adata_copy = adata.copy()

        sc.pp.pca(adata_copy, n_comps=10, svd_solver="arpack")
        sc.external.pp.harmony_integrate(adata_copy, key="orig.ident")
        sc.pp.neighbors(adata_copy, n_neighbors=10, n_pcs=10, use_rep="X_pca_harmony")

        best_res = 0
        best_score = -1
        best_labels = pd.Series(["X"] * adata_copy.n_obs)

        # Resolution range.
        for res in np.linspace(0.05, 2.0, 40):

            # Leiden Clustering.
            sc.tl.leiden(adata_copy, resolution=0.1, flavor="igraph", n_iterations=2)
            if adata_copy.obs["leiden"].nunique() < 2:
                continue
            new_score = silhouette_score(adata_copy.X, adata_copy.obs["leiden"], metric='euclidean', sample_size=1000)

            # New Score.
            if new_score > best_score:
                best_score = new_score
                best_res = res
                best_labels = adata_copy.obs["leiden"].copy()
            else:
                print(f"Resolution {res:.2f} did not improve silhouette score: {new_score:.4f}")
                return res, best_labels
        
        return best_res, best_labels

    @staticmethod
    def get_prefix_table(adata, column):
        prefix_table = defaultdict(list)
        for x in adata.obs[column].dropna().unique():
            prefix = ".".join(x.split(".")[:-1])
            prefix_table[prefix].append(x)
        return prefix_table

    @staticmethod
    def permutation_testing(X, y, n_permutations=20, shuffle=True):
        model = SGDClassifier(class_weight="balanced", max_iter=1000, random_state=42)
        permutation_accuracies = []
        for i in range(n_permutations):

            if shuffle:
                # Shuffle the labels
                y_permuted = np.random.permutation(y)
            
            else:
                # Use the original labels
                y_permuted = y

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_permuted,
                stratify=y_permuted,
                test_size=0.8,
            )

            model.fit(X_train, y_train)
            perm_accuracy = model.score(X_test, y_test)
            permutation_accuracies.append(perm_accuracy)

        return np.array(permutation_accuracies)

    def run(self, sample_id = "orig.ident"):

        _, labels = self.optimal_clustering(self.adata)
        self.adata.obs["choir_level_0"] = labels

        # Setup the initial clusters.
        for i in trange(1, 6):
            self.adata.obs[f"choir_level_{i}"] = None
            print(f"Running choir_level_{i}...")
            for cluster in self.adata.obs[f"choir_level_{i-1}"].unique():
                if cluster is None:
                    continue

                adata_subset = self.adata[self.adata.obs[f"choir_level_{i-1}"] == cluster].copy()

                if adata_subset.n_obs < 100:
                    continue

                sc.pp.pca(adata_subset, n_comps=10, svd_solver="arpack")
                if sample_id != None:
                    sc.external.pp.harmony_integrate(adata_subset, key="orig.ident")
                    sc.pp.neighbors(adata_subset, n_neighbors=10, n_pcs=10, use_rep="X_pca_harmony")
                else:
                    sc.pp.neighbors(adata_subset, n_neighbors=10, n_pcs=10, use_rep="X_pca")

                best_res, labels = self.optimal_clustering(adata_subset)

                self.adata[self.adata.obs[f"choir_level_{i-1}"] == cluster].obs[f"choir_level_{i}"] = labels

        for i in list(range(1, 5))[::-1]:
            print(f"Attempting to merge choir_level_{i+1}...")
            prefix_table = self.get_prefix_table(adata=self.adata, column=f"choir_level_{i+1}")

            for parent, children in tqdm(prefix_table.items()):
                if len(children) < 2:
                    continue

                X = self.adata[self.adata.obs[f"choir_level_{i+1}"].isin(children)].X
                y = self.adata[self.adata.obs[f"choir_level_{i+1}"].isin(children)].obs[f"choir_level_{i+1}"]

                if len(y.unique()) < 2:
                    continue

                true_accuracy = pd.Series(self.permutation_testing(X, y, shuffle=False))
                bootstrapped_null = pd.Series(self.permutation_testing(X, y))

                statistical_test = scipy.stats.ttest_ind(true_accuracy, bootstrapped_null, equal_var=False, nan_policy="omit")

                if statistical_test.pvalue > 0.05:
                    self.adata[self.adata.obs[f"choir_level_{i+1}"].isin(children)].obs[f"choir_level_{i+1}"] = None