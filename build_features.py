import trial_sets
import data
import gc


dl = data.DataLoader(window_size=100, threshold=1.0, algo_name='enhanced', features='comprehensive')

for trial_id in trial_sets.top_ids:
    print("Building cache for trial " + str(trial_id))
    X, y = dl.load([trial_id])
    del X
    del y
    gc.collect()
