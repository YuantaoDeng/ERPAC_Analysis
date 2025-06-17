import os
import pickle
import numpy as np
from hdf5storage import loadmat, savemat
from collections import defaultdict



DATA_DIR    = r"F:\Data\P12_ERPAC_Retrieval_window"
OUTPUT_MAT  = "grouped_data.mat"
OUTPUT_PKL  = "grouped_data.pkl"



grouped_data = defaultdict(list)

for subj_id in range(1, 21):
    file_path = os.path.join(DATA_DIR, f"subj_{subj_id}.mat")
    if not os.path.isfile(file_path):
        print(f"[Skip] File not exist: {file_path}")
        continue

    try:
        mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"[Error] when reading {file_path}: {e}")
        continue

    cs = mat.get('combinedStruct')
    if cs is None:
        print(f"[Warning] {file_path} lack combinedStruct")
        continue

    if hasattr(cs, '_fieldnames'):
        entries = [cs]
    elif isinstance(cs, np.ndarray):
        entries = cs.flat
    else:
        print(f"[Warning] combinedStruct type unknown: {type(cs)}")
        continue

    # divide as  (Location, Location_2)
    for entry in entries:
        loc1 = getattr(entry, 'Location', None)
        loc2 = getattr(entry, 'Location_2', None)
        if loc1 is None or loc2 is None:
            continue

        grouped_data[(loc1, loc2)].append({
            'SubjectID':               subj_id,
            'Electrode':               entry.Electrode,
            'Electrode_2':             entry.Electrode_2,
            'Location':                loc1,
            'Location_2':              loc2,
            'Comodulogram_match':      entry.Comodulogram_match,
            'Comodulogram_mismatch':   entry.Comodulogram_mismatch
        })


# saved as MATLAB v7.3 HDF5 file
mat_grouped = {}
for (loc1, loc2), entries in grouped_data.items():
    key_str = f"{loc1}__{loc2}"
    mat_grouped[key_str] = entries

savemat(OUTPUT_MAT, {'grouped_data': mat_grouped}, format='7.3')

# saved as Python pickle (.pkl)
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(grouped_data, f)

print(f"Dividing complete, saved to:\n - {OUTPUT_MAT}\n - {OUTPUT_PKL}")
