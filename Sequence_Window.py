import os
import pickle
import numpy as np
from scipy.io import loadmat


DATA_DIR      = r"F:\Data\P12_ERPAC_Retrieval_window"
GROUPED_PKL   = "grouped_data.pkl"
RESULT_FILE   = r"F:\Data\meanRT.mat"
OUTPUT_PKL    = "windowed_data.pkl"
PRE_WINDOW    = 0.4
POST_WINDOW   = 0.4


res      = loadmat(RESULT_FILE, squeeze_me=True)
meanRTs  = res.get('rt', res.get('meanRT'))
meanRTs  = np.asarray(meanRTs).flatten()  # shape = (20,)


with open(GROUPED_PKL, 'rb') as f:
    grouped_data = pickle.load(f)


samples = []
for entries in grouped_data.values():
    for entry in entries:
        subj    = entry['SubjectID']
        meanrt  = meanRTs[subj - 1]

        for key_label, label in [
            ('Comodulogram_match',    1),
            ('Comodulogram_mismatch', 0)
        ]:
            arr = entry[key_label]    # shape = (2,2,T_i)
            T   = arr.shape[2]

            # arr original time range [-0.4, meanrt+1.0]，Total duration Ts = meanrt + 1.4
            total_duration = meanrt + 1.0 + 0.4
            dt             = total_duration / (T - 1)

            # Calculate the index of RT at arr
            idx_rt = int(round((meanrt + 0.4) / dt))
            # calculate # of points at the window's edge
            pre_pts  = int(round(PRE_WINDOW  / dt))
            post_pts = int(round(POST_WINDOW / dt))

            start = idx_rt - pre_pts
            end   = idx_rt + post_pts

            # Edge check
            if start < 0 or end > T:
                continue

            window = arr[:, :, start:end]  # (2,2,window length)
            samples.append({
                'SubjectID': subj,
                'Location':  entry['Location'],
                'Location_2':entry['Location_2'],
                'label':      label,
                'window':     window
            })


# Find the maximum length
max_len = max(w['window'].shape[2] for w in samples)

def resample_window(arr: np.ndarray, N: int) -> np.ndarray:
    """
    From (2,2,L) to (2,2,N)
    """
    L = arr.shape[2]
    if L == N:
        return arr
    orig_idx   = np.linspace(0, L - 1, L)
    target_idx = np.linspace(0, L - 1, N)
    out = np.zeros((2, 2, N), dtype=arr.dtype)
    for i in range(2):
        for j in range(2):
            out[i, j, :] = np.interp(target_idx, orig_idx, arr[i, j, :])
    return out

for s in samples:
    s['window'] = resample_window(s['window'], max_len)

# Save the result
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(samples, f)

print(f"{len(samples)} windows，length = {max_len} points")
print(f"Result saved as: {OUTPUT_PKL}")
