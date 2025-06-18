import pickle
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import wilcoxon


algorithm = 'dtw'
baseline = 0.5                # baseline(In this case 0.5)
pkl_file = f"results_{algorithm}.pkl"
output_csv = f"accuracy_analysis_{algorithm}.csv"


# Load results dict
with open(pkl_file, 'rb') as f:
    results = pickle.load(f)


rows = []
for (loc1, loc2), info in results.items():
    n = info['n_samples']
    scores = np.array(info['scores'])      # shape = (n_splits,)
    mean_acc = scores.mean()
    successes = int(round(mean_acc * n))
    # Wilson CI
    ci_low, ci_high = proportion_confint(successes, n, method='wilson')

    rows.append({
        'Location':     loc1,
        'Location_2':   loc2,
        'n_samples':    n,
        'mean_accuracy':mean_acc,
        'ci_lower':     ci_low,
        'ci_upper':     ci_high
    })

df = pd.DataFrame(rows)

# Wilcoxon signed-rank test
#    test the difference between mean_accuracy and baseline
diffs = df['mean_accuracy'] - baseline
stat, p_value = wilcoxon(diffs)
print(f"Wilcoxon signed‐rank test vs. baseline={baseline}:")
print(f"  statistic = {stat:.3f}, p‐value = {p_value:.3e}")

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"Saved per‐group accuracy analysis to '{output_csv}'")
