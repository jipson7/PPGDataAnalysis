import trial_sets
import data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib2tikz import save as tikz_save
from collections import Counter
from data import GRAPH_CACHE

def bland_altman(data1, data2, *args, **kwargs):
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    print(md)
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    print(sd)

    count = Counter(zip(mean, diff))
    x, y = zip(*count.keys())
    s = np.array(list(count.values()))
    s = np.sqrt(s)*1.2

    fig = plt.figure(figsize=(4 * 1.2, 3 * 1.2))
    ax = fig.add_subplot(111)
    #ax.scatter(mean + x_noise, diff + y_noise, *args, **kwargs)
    ax.scatter(x,y,s=s,*args,*kwargs)
    ax.axhline(md,           color='gray', linestyle='--')
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax.set_ylabel('Difference between methods')
    ax.set_xlabel('Mean of methods')

    trans = transforms.blended_transform_factory(
        ax.transAxes, ax.transData)
    limitOfAgreement = 1.96

    limitOfAgreementRange = (md + (limitOfAgreement * sd)) - (md - limitOfAgreement * sd)
    offset = (limitOfAgreementRange / 100.0) * 1.5

    ax.text(0.02, md + offset, 'Mean', ha="left", va="bottom", transform=trans)
    ax.text(0.02, md - offset, f'{md:.2f}', ha="left", va="top", transform=trans)

    ax.text(0.02, md + (limitOfAgreement * sd) + offset, f'+{limitOfAgreement:.2f} SD', ha="left", va="bottom",
            transform=trans)
    ax.text(0.02, md + (limitOfAgreement * sd) - offset, f'{md + limitOfAgreement*sd:.2f}', ha="left", va="top",
            transform=trans)

    ax.text(0.02, md - (limitOfAgreement * sd) - offset, f'-{limitOfAgreement:.2f} SD', ha="left", va="top",
            transform=trans)
    ax.text(0.02, md - (limitOfAgreement * sd) + offset, f'{md - limitOfAgreement*sd:.2f}', ha="left", va="bottom",
            transform=trans)

    return fig

# dl = data.DataLoader(window_size=100, threshold=2.0, algo_name='enhanced', features='comprehensive')
#
# reflective_valid = []
# transitive_valid = []
#
# ROUND_NUM = True
#
# for trial_id in trial_sets.top_ids:
#     print("Running trial " + str(trial_id))
#     _, reflective, transitive = dl.load_all_oxygen(trial_id)
#     for o1, o2 in zip(reflective, transitive):
#         if np.isnan(o1) or np.isnan(o2):
#             continue
#         if ROUND_NUM:
#             o1 = int(round(o1))
#             o2 = int(round(o2))
#         reflective_valid.append(o1)
#         transitive_valid.append(o2)
#
# reflective_valid = np.array(reflective_valid)
# transitive_valid = np.array(transitive_valid)

import pickle

# pickle.dump((reflective_valid, transitive_valid), open("daniyals_pickle.pkl", "wb"))

reflective_valid, transitive_valid = pickle.load(open("daniyals_pickle.pkl", "rb"))

bland_altman(reflective_valid, transitive_valid)

# plt.show()
plt.tight_layout()
plt.savefig(GRAPH_CACHE + 'bland-altman.pdf')
plt.show()
