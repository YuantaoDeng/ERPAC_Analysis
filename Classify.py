import pickle
from collections import defaultdict
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score

# DTW
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
# BOSS
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.compose import ColumnEnsembleClassifier
# ROCKET
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

# Choose algorithm：'dtw', 'boss', 'rocket'
algorithm = 'rocket'

print(f"Algorithm：{algorithm.upper()}")

# Load windowed_data.pkl
with open('windowed_data.pkl', 'rb') as f:
    samples = pickle.load(f)

# Divide as (Location, Location_2)
grouped = defaultdict(list)
for s in samples:
    grouped[(s['Location'], s['Location_2'])].append(s)

results   = {}
min_samples = 20   # skip threshold

for key, group_samples in grouped.items():
    n = len(group_samples)
    if n < min_samples:
        print(f"Skip group {key}，number of samples: {n} < {min_samples}")
        continue

    # Construct X_np and y
    N    = group_samples[0]['window'].shape[2]
    X_np = np.zeros((n, N, 4))  # (n_samples, time_length, n_channels)
    y    = np.zeros(n, dtype=int)
    for i, s in enumerate(group_samples):
        arr     = s['window']           # (2,2,N)
        X_np[i] = arr.reshape(4, N).T
        y[i]    = s['label']


    if algorithm == 'dtw':
        X_input = X_np
        clf     = KNeighborsTimeSeriesClassifier(
            n_neighbors=1, metric="dtw", n_jobs=-1
        )
    elif algorithm == 'rocket':
        X_input = X_np
        rocket = Rocket(num_kernels=10000, random_state=42)
        ridge  = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        clf     = make_pipeline(rocket, ridge)
    elif algorithm == 'boss':
        col_names = [f"ch{c}" for c in range(4)]
        X_input = pd.DataFrame({
            col: [pd.Series(X_np[i, :, idx]) for i in range(n)]
            for idx, col in enumerate(col_names)
        })
        estimators = [
            (col, BOSSEnsemble(random_state=42), [col])
            for col in col_names
        ]
        clf = ColumnEnsembleClassifier(estimators=estimators)
    else:
        raise ValueError(f"Algorithm not supported: {algorithm}")

    # 5-fold cross_val
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        clf, X_input, y,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )

    results[key] = {'n_samples': n, 'scores': scores}
    print(f"{key}: Number of samples ={n}, Accuracy ={scores.mean():.3%} ± {scores.std():.3%}")


print("\nConclusion")
for key, info in results.items():
    m, s = info['scores'].mean(), info['scores'].std()
    print(f"{key}: Average accuracy={m:.3%}, Std={s:.3%}")

total_n     = sum(info['n_samples'] for info in results.values())
weighted_acc = sum(info['scores'].mean() * info['n_samples']
                   for info in results.values()) / total_n
print(f"\nWeighted accuracy：{weighted_acc:.3%}")
