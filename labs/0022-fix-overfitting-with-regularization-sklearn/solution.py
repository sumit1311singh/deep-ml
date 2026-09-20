import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error
# You can import any sklearn module you need

def train(X_train, y_train, X_val, y_val):
    """
    Train a regression model that generalizes well despite having
    MORE features than training samples (many are noise or redundant).
    
    WARNING: LinearRegression() WILL overfit here.
    - LinearRegression(): Train R² ≈ 1.0, Val R² ≈ -5.0
    - You need regularization to pass!
    
    Args:
        X_train: numpy array of shape (n_samples, n_features) -- standardized
                 (~250 samples, ~264 features -- more features than samples!)
        y_train: numpy array of shape (n_samples,) -- target values
        X_val:   numpy array of shape (n_val, n_features) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation targets
    
    Returns:
        predict: callable that takes X (n, n_features) and returns y_pred (n,)
    """
    # TODO: fit a regularized model and return a predict function
    # 
    # Ideas:
    #   - Ridge(alpha=??) or RidgeCV (auto-tunes alpha)
    #   - Lasso(alpha=??) or LassoCV (auto-tunes + feature selection)
    #   - ElasticNet (combines L1 + L2)
    #   - SelectKBest + Ridge pipeline
    #   - Any sklearn approach that handles overfitting!
    #
    # Plain LinearRegression() will NOT work here.
    
    models = [
        (RidgeCV, {"alphas": [0.1, 1, 10, 100, 300, 1000]}),
        (LassoCV, {"alphas": [0.001, 0.01, 0.1, 1, 10]}),
        (ElasticNetCV, {"alphas": [0.01, 0.1, 1, 10], "l1_ratio": [0.1, 0.5, 0.7, 0.9, 1.0]})
    ]

    best_mse = float("inf")
    best_model = None

    for model_type, params in models:
        model = model_type(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)

        if mse < best_mse:
            best_mse = mse
            best_model = model

    def predict(X):
        return best_model.predict(X)

    return predict
