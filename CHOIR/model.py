import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
from IPython.display import clear_output
from sklearn.model_selection import train_test_split

def model_selection(model_type="ensemble"):
    """
    Selects and returns a machine learning model based on the specified model_type.

    Parameters
    ----------
    model_type : str or object, default="ensemble"
        The type of model to select. Supported string values are:
            - "ensemble": RandomForestClassifier from sklearn.ensemble
            - "linear_model": LogisticRegression from sklearn.linear_model
            - "neural_network": RandomForestClassifier from sklearn.ensemble (placeholder)
        Alternatively, a custom model object can be provided if it implements
        'fit', 'score', and 'predict' methods.

    Returns
    -------
    model : object
        An instance of the selected model.

    Raises
    ------
    ValueError
        If the model_type is not recognized or does not have the required methods.
    """
    # Model Type Selection
    if isinstance(model_type, str):
        print(f"Model type: {model_type}")
        if model_type == "ensemble":
            model = sklearn.ensemble.RandomForestClassifier(class_weight="balanced")
        if model_type == "linear_model":
            model = sklearn.linear_model.LogisticRegression(class_weight="balanced")
        if model_type == "neural_network":
            model = sklearn.ensemble.RandomForestClassifier(class_weight="balanced")
        return model

    # Duck typing: check if model_type has fit, score, and predict.
    elif all(hasattr(model_type, attr) for attr in ["fit", "score", "predict"]):
        model = model_type
        return model

    else:
        raise ValueError(f"Model type '{model_type}' is not recognized or does not have the required methods.")


def optimal_cluster(adata, resolution_range=(0.10, 2.0, 0.10), alpha=0.95, model_type = "linear_model", test_size = 0.5):
    """
    Find the optimal clustering resolution for the given AnnData object.
    
    Parameters:
    adata (AnnData): The annotated data matrix.
    resolution_range (tuple): A tuple defining the range of resolutions to test.
    alpha (float): Hypothesis testing alpha by bootstrapped estimators.
    test_size (float): Test size when performing test-train split (fraction of the population).
    
    Returns:
    None: The function modifies the adata object in place.
    """
    # Fitted Labels
    new_labels = None

    adata_copy = adata.copy()

    # If there user has selected highly variable genes, filter the data.
    if "highly_variable" in adata.var.columns:
        adata_copy = adata_copy[:, adata_copy.var["highly_variable"]]

    # Create a copy.
    X = adata_copy.X
    y = None

    model = model_selection(model_type)

    # Iterative clustering.
    for res in np.arange(resolution_range[0], resolution_range[1] + resolution_range[1], resolution_range[2]):
        # Fuck rounding errors.
        res = round(res, 2)

        # Generate clusters.
        sc.tl.leiden(adata_copy, resolution=res, flavor='igraph', n_iterations=2)

        if adata_copy.obs['leiden'].nunique() == 1:
            continue

        y = adata_copy.obs['leiden']

        # Get the labels, test train split.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        if accuracy > alpha:
            new_labels = y

        else:
            clear_output(wait=True)
            print(f"Resolution {res} failed the stability test. Accuracy = {round(accuracy, 3)}")
            return round(res - resolution_range[2], 2), new_labels

    return round(res, 2), new_labels