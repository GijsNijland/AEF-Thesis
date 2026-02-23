# =========================================================
# Regularized_HARX.py
# Paper-accurate HAR-X regularization (RR, LA, EN, A-LA, P-LA)
# =========================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# ---------------------------------------------------------
# Import required data-prep functions from earlier files
# ---------------------------------------------------------
from HARX_regressions import get_harx_data
from HAR_regressions import mse

# Load all HAR-X regressors (extended predictor set)
X, y = get_harx_data()


# =========================================================
# Small helper: safe scalar prediction extraction
# =========================================================
def scalar_pred(model, X2D):
    """Safely extract a scalar prediction from sklearn .predict() output."""
    yhat = model.predict(X2D)
    if hasattr(yhat, "values"):
        yhat = yhat.values
    arr = np.asarray(yhat).ravel()
    return float(arr[0])


# =========================================================
# Helper: Tune λ (and α for EN) exactly as in the paper
# =========================================================
def tune_regularization(model_class, X_train, y_train, X_val, y_val):
    lambdas = np.logspace(-5, 2, 100)     # λ-grid
    alphas  = np.linspace(0, 1, 10)       # EN only

    best_mse = np.inf
    best_model = None
    best_scaler = None

    # Standardize X only
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train.astype(float))
    Xvl = scaler.transform(X_val.astype(float))

    for lam in lambdas:

        if model_class == ElasticNet:
            for a in alphas:
                model = ElasticNet(alpha=lam, l1_ratio=a, max_iter=5000)
                model.fit(Xtr, y_train)
                mse_val = ((model.predict(Xvl) - y_val) ** 2).mean()

                if mse_val < best_mse:
                    best_mse = mse_val
                    best_model = model
                    best_scaler = scaler

        else:
            model = model_class(alpha=lam, max_iter=5000)
            model.fit(Xtr, y_train)
            mse_val = ((model.predict(Xvl) - y_val) ** 2).mean()

            if mse_val < best_mse:
                best_mse = mse_val
                best_model = model
                best_scaler = scaler

    return best_model, best_scaler


# =========================================================
# Base rolling window (train + validation) for RR, LA, EN
# =========================================================
def rolling_regularized(model_class, X, y, train_size=0.7, val_size=0.1):
    T = len(X)

    train_end = int(train_size * T)
    val_end   = int((train_size + val_size) * T)

    preds = []
    actuals = []

    for t in range(val_end, T):

        X_train = X.iloc[:train_end].astype(float)
        y_train = y.iloc[:train_end]

        X_val   = X.iloc[train_end:val_end].astype(float)
        y_val   = y.iloc[train_end:val_end]

        X_test  = X.iloc[t:t+1].astype(float)

        # Tune λ (and α for EN)
        best_model, scaler = tune_regularization(model_class, X_train, y_train, X_val, y_val)

        Xt = scaler.transform(X_test)
        pred = scalar_pred(best_model, Xt)   # <-- FIXED

        preds.append(pred)
        actuals.append(y.iloc[t])

        train_end += 1
        val_end   += 1

    # Correct index: predictions start at first t = val_end
    idx = X.index[int((train_size + val_size) * T):]
    return pd.Series(preds, index=idx), pd.Series(actuals, index=idx)


# =========================================================
# Adaptive Lasso HAR-X (two-stage)
# =========================================================
def rolling_adaptive_lasso(X, y, train_size=0.7, val_size=0.1):
    T = len(X)

    train_end = int(train_size * T)
    val_end   = int((train_size + val_size) * T)

    preds = []
    actuals = []

    for t in range(val_end, T):

        X_train = X.iloc[:train_end].astype(float)
        y_train = y.iloc[:train_end]

        X_val   = X.iloc[train_end:val_end].astype(float)
        y_val   = y.iloc[train_end:val_end]

        X_test  = X.iloc[t:t+1].astype(float)

        # Stage 1: OLS
        Xc = sm.add_constant(X_train)
        ols = sm.OLS(y_train, Xc).fit()
        beta = ols.params[1:]
        weights = 1 / (np.abs(beta) + 1e-6)

        Xw_train = X_train * weights.values
        Xw_val   = X_val   * weights.values
        Xw_test  = X_test  * weights.values

        best_lasso, scaler = tune_regularization(Lasso, Xw_train, y_train, Xw_val, y_val)
        pred = scalar_pred(best_lasso, scaler.transform(Xw_test))  # <-- FIXED

        preds.append(pred)
        actuals.append(y.iloc[t])

        train_end += 1
        val_end   += 1

    idx = X.index[int((train_size + val_size) * T):]
    return pd.Series(preds, index=idx), pd.Series(actuals, index=idx)


# =========================================================
# Post-Lasso HAR-X (selection → OLS)
# =========================================================
def rolling_post_lasso(X, y, train_size=0.7, val_size=0.1):
    T = len(X)

    train_end = int(train_size * T)
    val_end   = int((train_size + val_size) * T)

    preds = []
    actuals = []

    for t in range(val_end, T):

        X_train = X.iloc[:train_end].astype(float)
        y_train = y.iloc[:train_end]

        X_val   = X.iloc[train_end:val_end].astype(float)
        y_val   = y.iloc[train_end:val_end]

        X_test  = X.iloc[t:t+1].astype(float)

        best_lasso, scaler = tune_regularization(Lasso, X_train, y_train, X_val, y_val)
        mask = best_lasso.coef_ != 0

        if mask.sum() == 0:
            pred = float(y_train.mean())
        else:
            Xs_train = X_train.iloc[:, mask]
            Xs_test  = X_test.iloc[:, mask]

            Xc = sm.add_constant(Xs_train)
            beta = np.linalg.lstsq(Xc, y_train, rcond=None)[0]

            pred = float(sm.add_constant(Xs_test) @ beta)

        preds.append(pred)
        actuals.append(y.iloc[t])

        train_end += 1
        val_end   += 1

    idx = X.index[int((train_size + val_size) * T):]
    return pd.Series(preds, index=idx), pd.Series(actuals, index=idx)


# =========================================================
#  RUN ALL REGULARIZED HAR-X MODELS
# =========================================================
if __name__ == "__main__":

    print("\n[RIDGE HAR-X]")
    rr_fc, rr_act = rolling_regularized(Ridge, X, y)
    print("MSE:", mse(rr_fc, rr_act))

    print("\n[LASSO HAR-X]")
    la_fc, la_act = rolling_regularized(Lasso, X, y)
    print("MSE:", mse(la_fc, la_act))

    print("\n[ELASTIC NET HAR-X]")
    en_fc, en_act = rolling_regularized(ElasticNet, X, y)
    print("MSE:", mse(en_fc, en_act))

    print("\n[ADAPTIVE LASSO HAR-X]")
    ala_fc, ala_act = rolling_adaptive_lasso(X, y)
    print("MSE:", mse(ala_fc, ala_act))

    print("\n[POST-LASSO HAR-X]")
    pl_fc, pl_act = rolling_post_lasso(X, y)
    print("MSE:", mse(pl_fc, pl_act))