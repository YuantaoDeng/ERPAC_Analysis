import pickle
from collections import defaultdict
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.dictionary_based import BOSSEnsemble

# Choose algorithm: 'dtw', 'boss', or 'rocket'
algorithm = 'dtw'

print(f"Algorithm: {algorithm.upper()}")

# Load preprocessed windows
with open('windowed_data.pkl', 'rb') as f:
    samples = pickle.load(f)

# Group samples by (Location, Location_2)
grouped = defaultdict(list)
for s in samples:
    grouped[(s['Location'], s['Location_2'])].append(s)

results = {}
min_samples = 20  # skip groups with fewer samples

# Iterate each group
for key, group_samples in grouped.items():
    n = len(group_samples)
    if n < min_samples:
        print(f"Skip group {key}, sample count {n} < {min_samples}")
        continue

    # 1) Build numpy array X_np: (n_samples, time_length, n_channels)
    N = group_samples[0]['window'].shape[2]
    X_np = np.zeros((n, N, 4))
    y    = np.zeros(n, dtype=int)
    for i, s in enumerate(group_samples):
        arr = s['window']             # (2,2,N)
        X_np[i] = arr.reshape(4, N).T
        y[i]    = s['label']

    # —— Flatten channels: treat each channel as an independent window —— #
    # X_flat: shape (n_samples*4, time_length, 1)
    X_flat = X_np.transpose(0,2,1).reshape(-1, N, 1)
    # y_flat: repeat each label 4 times
    y_flat = np.repeat(y, 4)

    # Prepare cross-validation on flattened data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    if algorithm == 'dtw':
        # DTW-KNN on each flattened window
        for train_idx, test_idx in cv.split(X_flat, y_flat):
            clf = KNeighborsTimeSeriesClassifier(
                n_neighbors=1, metric="dtw", n_jobs=-1
            )
            clf.fit(X_flat[train_idx], y_flat[train_idx])
            y_pred = clf.predict(X_flat[test_idx])
            fold_scores.append(accuracy_score(y_flat[test_idx], y_pred))

    elif algorithm == 'rocket':
        # RocketClassifier on nested DataFrame of single-channel series
        # build once outside CV
        X_series = pd.DataFrame({
            'ts': [pd.Series(X_flat[i, :, 0]) for i in range(X_flat.shape[0])]
        })
        for train_idx, test_idx in cv.split(X_series, y_flat):
            rocket = RocketClassifier(num_kernels=10000, random_state=42)
            rocket.fit(X_series.iloc[train_idx], y_flat[train_idx])
            y_pred = rocket.predict(X_series.iloc[test_idx])
            fold_scores.append(accuracy_score(y_flat[test_idx], y_pred))

    else:  # boss
        # BOSSEnsemble on nested DataFrame of single-channel series
        X_series = pd.DataFrame({
            'ts': [pd.Series(X_flat[i, :, 0]) for i in range(X_flat.shape[0])]
        })
        for train_idx, test_idx in cv.split(X_series, y_flat):
            bos = BOSSEnsemble(random_state=42)
            bos.fit(X_series.iloc[train_idx], y_flat[train_idx])
            y_pred = bos.predict(X_series.iloc[test_idx])
            fold_scores.append(accuracy_score(y_flat[test_idx], y_pred))

    # record results: now n_samples = original_n * 4
    scores = np.array(fold_scores)
    results[key] = {'n_samples': X_flat.shape[0], 'scores': scores}
    print(f"{key}: total_windows={X_flat.shape[0]}, accuracy={scores.mean():.3%} ± {scores.std():.3%}")

# Summary
print("\n=== Summary ===")
summary_rows = []
for key, info in results.items():
    m, s = info['scores'].mean(), info['scores'].std()
    print(f"{key}: mean accuracy={m:.3%}, std={s:.3%}")
    summary_rows.append({
        'Location':    key[0],
        'Location_2':  key[1],
        'n_windows':   info['n_samples'],
        'mean_accuracy': m,
        'std_accuracy':  s
    })

# Weighted overall accuracy
total_n     = sum(info['n_samples'] for info in results.values())
weighted_acc = sum(info['scores'].mean() * info['n_samples']
                   for info in results.values()) / total_n
print(f"\nWeighted overall accuracy: {weighted_acc:.3%}")

# Save full results and summary
output_pickle = f"results_{algorithm}.pkl"
with open(output_pickle, 'wb') as f:
    pickle.dump(results, f)
print(f"Saved full results dict to {output_pickle}")

summary_df = pd.DataFrame(summary_rows)
output_csv = f"summary_{algorithm}.csv"
summary_df.to_csv(output_csv, index=False)
print(f"Saved summary table to {output_csv}")
