import trial_sets
import data
import numpy as np
from pyCompare import blandAltman

dl = data.DataLoader(window_size=100, threshold=2.0, algo_name='enhanced', features='comprehensive')

reflective_valid = []
transitive_valid = []

ROUND_NUM = True

for trial_id in trial_sets.top_ids:
    print("Running trial " + str(trial_id))
    _, reflective, transitive = dl.load_all_oxygen(trial_id)
    for o1, o2 in zip(reflective, transitive):
        if np.isnan(o1) or np.isnan(o2):
            continue
        if ROUND_NUM:
            o1 = int(round(o1))
            o2 = int(round(o2))
        reflective_valid.append(o1)
        transitive_valid.append(o2)

reflective_valid = np.array(reflective_valid)
transitive_valid = np.array(transitive_valid)

savePath = 'fingertip-ba.png'

if ROUND_NUM:
    savePath = 'rounded-' + savePath

savePath = data.GRAPH_CACHE + savePath

blandAltman(reflective_valid, transitive_valid)

