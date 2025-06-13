import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import scanpy as sc
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class CHOIR:
    def __init__(self, adata, batch_key="orig.ident"):
        self.adata = adata
        self.batch_key = batch_key

    def model_selection(self, model_type="ensemble"):
        if isinstance(model_type, str):
            print(f"Model type: {model_type}")
            if model_type == "ensemble":
                return sklearn.ensemble.RandomForestClassifier(class_weight="balanced")
            if model_type == "linear_model":
                return sklearn.linear_model.LogisticRegression(class_weight="balanced")
            if model_type == "neural_network":
                return sklearn.neural_network.MLPClassifier(class_weight="balanced")
        elif all(hasattr(model_type, attr) for attr in ["fit", "score", "predict"]):
            return model_type
        else:
            raise ValueError(f"Model type '{model_type}' is not recognized or does not have the required methods.")

    def optimal_cluster(self, resolution_range=(0.05, 2.0, 0.05), alpha=0.95, model_type="linear_model", test_size=0.5):
        new_labels = None
        adata_copy = self.adata.copy()

        if "highly_variable" in adata_copy.var.columns:
            adata_copy = adata_copy[:, adata_copy.var["highly_variable"]]

        model = sklearn.linear_model.LogisticRegression(class_weight="balanced")
        model.fit(adata_copy.X, adata_copy.obs[self.batch_key])
        adata_copy = adata_copy[:, model.coef_.max(axis=0) < pd.Series(model.coef_.max(axis=0)).quantile(.90)]

        sc.tl.pca(adata_copy, svd_solver="arpack")

        if self.batch_key in adata_copy.obs.columns and adata_copy.obs[self.batch_key].nunique() > 1:
            sc.external.pp.harmony_integrate(adata_copy, key=self.batch_key)
        else:
            print(f"Skipping Harmony: insufficient metadata in '{self.batch_key}'")

        sc.pp.neighbors(adata_copy, n_neighbors=20, n_pcs=20, use_rep="X_pca_harmony")

        model = self.model_selection(model_type)

        X = adata_copy.X
        for res in np.arange(resolution_range[0], resolution_range[1] + resolution_range[2], resolution_range[2]):
            res = round(res, 2)
            sc.tl.leiden(adata_copy, resolution=res, flavor='igraph', n_iterations=2)

            if adata_copy.obs['leiden'].nunique() == 1:
                continue

            y = adata_copy.obs['leiden']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            per_class_accuracies = [
                accuracy_score(y_test[y_test == label], y_pred[y_test == label])
                for label in y_test.cat.categories if (y_test == label).sum() > 0
            ]

            min_acc = min(per_class_accuracies)
            if min_acc >= alpha:
                new_labels = y
            else:
                clear_output(wait=True)
                print(f"Resolution {res} failed: min class accuracy = {round(min_acc, 3)}")
                return round(res - resolution_range[2], 2), new_labels

        return round(res, 2), new_labels

    def run_clustering_hierarchy(self, alpha=0.9):
        res, labels = self.optimal_cluster(alpha=alpha)
        self.adata.obs["choir_level_0"] = labels

        for level in range(3):
            col = f"choir_level_{level + 1}"
            self.adata.obs[col] = self.adata.obs[f"choir_level_{level}"]

            for parent_label in self.adata.obs[f"choir_level_{level}"].cat.categories:
                if parent_label[-1] != "X":
                    indexer = self.adata.obs[f"choir_level_{level}"] == parent_label

                    if indexer.sum() < 100:
                        continue

                    res, labels = self.optimal_cluster(
                        resolution_range=(0.1, 2.0, 0.1),
                        alpha=0.7,
                        model_type="linear_model",
                        test_size=0.5
                    )

                    if labels is None:
                        new_labels = [f"{parent_label}.X"] * indexer.sum()
                    else:
                        new_labels = [f"{parent_label}.{lbl}" for lbl in labels]

                    new_cats = pd.unique(new_labels)
                    self.adata.obs[col] = self.adata.obs[col].cat.add_categories(new_cats)
                    self.adata.obs.loc[indexer, col] = new_labels
