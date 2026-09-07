"""Train and evaluate the feasibility model; the source of REPORT.md's numbers.

Three stages:
1. Spatial cross-validation (GroupKFold over 0.25-degree grid cells, so a
   fold never trains and tests on neighboring sites): AUC and P@100 for
   the gradient-boosting model and the logistic companion, plus a paired
   bootstrap on their AUC difference.
2. Final fit on all rows with isotonic calibration, saved to
   data/feasibility_model.joblib as {"model", "features"}.
3. Reproduction check: if a previous bundle exists, compare its
   predictions to the new one's (should agree to numerical noise).

Run: uv run python -m b2p.train_model
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BUNDLE = "data/feasibility_model.joblib"
FEATURES = [
    "elev_m", "relief_m", "valley_depth_m", "local_slope_deg",
    "area_slope_deg", "gap_width_m", "roughness_m",
    "clay_pct", "sand_pct", "bulk_density", "bedrock_depth_cm",
    "log_upstream", "hand_m", "channel_slope",
]


def boosting():
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=30, random_state=0)


def logistic():
    # Scaling matters here: regularized LR penalizes coefficients, and
    # unscaled features (meters vs degrees vs log km2) would make the
    # penalty arbitrary. Trees are invariant to this; LR is not.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def main():
    df = pd.read_csv("data/rwanda_features.csv")
    df["log_upstream"] = np.log10(df.upstream_km2 + 0.01)
    d = df.dropna(subset=FEATURES + ["label"])
    X, y = d[FEATURES].values, d.label.values
    groups = ((d.lat // 0.25).astype(int).astype(str) + "_"
              + (d.lon // 0.25).astype(int).astype(str))
    print(f"n={len(d)}  base rate={y.mean():.2f}  "
          f"spatial groups={groups.nunique()}")

    cv = GroupKFold(5)
    p_gb = cross_val_predict(boosting(), X, y, cv=cv, groups=groups,
                             method="predict_proba")[:, 1]
    p_lr = cross_val_predict(logistic(), X, y, cv=cv, groups=groups,
                             method="predict_proba")[:, 1]
    for name, p in [("boosting", p_gb), ("logistic", p_lr)]:
        top = np.argsort(-p)
        print(f"{name}: AUC={roc_auc_score(y, p):.3f}  "
              f"P@100={y[top[:100]].mean():.2f}  "
              f"P@200={y[top[:200]].mean():.2f}")

    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        diffs.append(roc_auc_score(y[i], p_gb[i]) - roc_auc_score(y[i], p_lr[i]))
    diffs = np.array(diffs)
    print(f"boosting - logistic AUC: {diffs.mean():+.3f}, "
          f"CI [{np.percentile(diffs, 2.5):.3f}, {np.percentile(diffs, 97.5):.3f}], "
          f"P(>0)={np.mean(diffs > 0):.2f}")

    model = CalibratedClassifierCV(boosting(), method="isotonic", cv=3)
    model.fit(X, y)

    try:
        old = joblib.load(BUNDLE)
        p_old = old["model"].predict_proba(d[old["features"]].values)[:, 1]
        p_new = model.predict_proba(X)[:, 1]
        print(f"reproduction check vs existing bundle: "
              f"max |dP|={np.abs(p_new - p_old).max():.4f}, "
              f"mean |dP|={np.abs(p_new - p_old).mean():.4f}")
    except FileNotFoundError:
        print("no existing bundle; skipping reproduction check")

    joblib.dump({"model": model, "features": FEATURES}, BUNDLE)
    print(f"saved {BUNDLE}")


if __name__ == "__main__":
    main()
