Modules

 Data_process  
- Reads in per-subject `.mat` files (e.g. `subj_1.mat … subj_20.mat`).  
- Extracts and groups relevant fields (`Location`, `Location_2`, `Comodulogram_match`, `Comodulogram_mismatch`).  
- Saves a consolidated `grouped_data.pkl` for downstream processing.

 Sequence_window  
- Loads `grouped_data.pkl` and subject reaction times (`meanRT.mat`).  
- For each trial window, cuts out a window from RT – 0.4 s to RT + 0.4 s.  
- Uses linear interpolation to resample each window to a common length `N`.  
- Outputs `windowed_data.pkl` containing uniform `(2,2,N)` arrays with associated labels.

 Classify  
- Loads `windowed_data.pkl`, flattens each `(2,2,N)` window into four time-series channels.  
- Supports three algorithms, selectable via a command-line or config flag:
  - DTW (1-NN): stable performance, fast to set up.  
  - BOSS Ensemble: highest accuracy, but slowest runtime.  
  - ROCKET + Ridge: best overall balance of speed and accuracy; some classes may underperform.  
- Performs per-location (“`Location, Location_2`”) 5-fold cross-validation, then reports mean ± std accuracy and a weighted overall accuracy.

---

 Algorithm Summaries

 1. DTW (Dynamic Time Warping) + 1-NN  
- Core idea: Measures elastic distance between two sequences, aligns them in time to compute similarity.  
- Pros: Robust to local time shifts; minimal hyperparameters.  
- Cons: Distance computations scale quadratically with sequence length.

 2. BOSS (Bag-Of-SFA-Symbols) Ensemble  
- Core idea:  
  1. Extract sliding-window subsequences.  
  2. Transform each window via Symbolic Fourier Approximation (SFA) into a discrete “word.”  
  3. Build a histogram of words (with numerosity reduction).  
  4. Classify via 1-NN over histograms; ensemble over multiple window lengths.  
- Pros: State-of-the-art accuracy on many benchmarks.  
- Cons: Heavy memory use and long training times; originally supports only univariate series.

、3. ROCKET (RandOm Convolutional KErnel Transform) + Ridge  
- Core idea:  
  1. Convolve each multivariate series with thousands of fixed random kernels.  
  2. Apply global pooling (max, proportion positive) to each feature map.  
  3. Feed the resulting features into a linear classifier (Ridge).  
- Pros: Extremely fast featurization; high accuracy with minimal tuning.  
- Cons: May underfit small classes; output feature dimension can be large.

---

 Usage

```bash
# 1. Data loading
python Data_process.py

# 2. Sequence extraction & interpolation
python Sequence_window.py \
  --input grouped_data.pkl \
  --meanrt meanRT.mat \
  --output windowed_data.pkl \
  --pre 0.4 --post 0.4

# 3. Classification
python Classify.py \
  --input windowed_data.pkl \
  --algorithm dtw
# or --algorithm boss
# or --algorithm rocket
