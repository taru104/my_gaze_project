import numpy as np
d = np.load(r'C:\Users\hazib\my_gaze_project\cache\sota_7d_checkpoint.npz')
print(f"Checkpoint frames : {len(d['X'])}")
print(f"X shape           : {d['X'].shape}")
print(f"Subjects          : {len(set(d['subj_id'].tolist()))}")
print(f"X[0] sample       : {d['X'][0]}")
