import pickle
from collections import defaultdict
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.compose import ColumnEnsembleClassifier

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

    # Build numpy array X_np: (n_samples, time_length, n_channels)
    N = group_samples[0]['window'].shape[2]
    X_np = np.zeros((n, N, 4))
    y = np.zeros(n, dtype=int)
    for i, s in enumerate(group_samples):
        arr = s['window']             # (2,2,N)
        X_np[i] = arr.reshape(4, N).T
        y[i] = s['label']

    # Build nested DataFrame X_df for rocket and boss
    col_names = [f'ch{c}' for c in range(4)]
    X_df = pd.DataFrame({
        col: [pd.Series(X_np[i, :, idx]) for i in range(n)]
        for idx, col in enumerate(col_names)
    })

    # Prepare cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    # Manual CV for dtw and rocket
    if algorithm in ('dtw', 'rocket'):
        for train_idx, test_idx in cv.split(X_np, y):
            # collect channel-wise predictions
            channel_preds = np.zeros((len(test_idx), 4), dtype=int)

            for ch in range(4):
                if algorithm == 'dtw':
                    # reshape to (n_samples, series_length, 1)
                    X_train_ch = X_np[train_idx, :, ch].reshape(-1, N, 1)
                    X_test_ch = X_np[test_idx, :, ch].reshape(-1, N, 1)
                    clf_ch = KNeighborsTimeSeriesClassifier(
                        n_neighbors=1, metric="dtw", n_jobs=-1
                    )
                    clf_ch.fit(X_train_ch, y[train_idx])
                    preds_ch = clf_ch.predict(X_test_ch)

                else:  # rocket
                    # use RocketClassifier on one-channel nested DataFrame
                    col = col_names[ch]
                    rocket = RocketClassifier(num_kernels=10000, random_state=42)
                    X_train_df = X_df.iloc[train_idx][[col]]
                    X_test_df = X_df.iloc[test_idx][[col]]
                    rocket.fit(X_train_df, y[train_idx])
                    preds_ch = rocket.predict(X_test_df)

                channel_preds[:, ch] = preds_ch

            # majority vote across channels
            votes = []
            for row in channel_preds:
                # choose most common label among four channels
                values, counts = np.unique(row, return_counts=True)
                votes.append(values[np.argmax(counts)])
            y_pred = np.array(votes)
            fold_scores.append(accuracy_score(y[test_idx], y_pred))

    else:
        # Use ColumnEnsembleClassifier for BOSS
        estimators = [
            (col, BOSSEnsemble(random_state=42), [col])
            for col in col_names
        ]
        clf = ColumnEnsembleClassifier(estimators=estimators)
        for train_idx, test_idx in cv.split(X_df, y):
            clf.fit(X_df.iloc[train_idx], y[train_idx])
            y_pred = clf.predict(X_df.iloc[test_idx])
            fold_scores.append(accuracy_score(y[test_idx], y_pred))

    # record results
    scores = np.array(fold_scores)
    results[key] = {'n_samples': n, 'scores': scores}
    print(f"{key}: samples={n}, accuracy={scores.mean():.3%} ± {scores.std():.3%}")

# Summary
print("\n=== Summary ===")
summary_rows = []
for key, info in results.items():
    m, s = info['scores'].mean(), info['scores'].std()
    print(f"{key}: mean accuracy={m:.3%}, std={s:.3%}")
    summary_rows.append({
        'Location': key[0],
        'Location_2': key[1],
        'n_samples': info['n_samples'],
        'mean_accuracy': m,
        'std_accuracy': s
    })

# Weighted overall accuracy
total_n = sum(info['n_samples'] for info in results.values())
weighted_acc = sum(info['scores'].mean() * info['n_samples']
                   for info in results.values()) / total_n
print(f"\nWeighted overall accuracy: {weighted_acc:.3%}")


output_pickle = f"results_{algorithm}.pkl"
with open(output_pickle, 'wb') as f:
    pickle.dump(results, f)
print(f"Saved full results dict to {output_pickle}")


summary_df = pd.DataFrame(summary_rows)
# convert accuracy to percentage strings if you like, or keep floats
# summary_df['mean_accuracy'] = summary_df['mean_accuracy'].map("{:.3%}".format)
# summary_df['std_accuracy']  = summary_df['std_accuracy'].map("{:.3%}".format)

output_csv = f"summary_{algorithm}.csv"
summary_df.to_csv(output_csv, index=False)
print(f"Saved summary table to {output_csv}")