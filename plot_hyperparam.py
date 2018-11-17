
import matplotlib.pyplot as plt
from data import GRAPH_CACHE, LTX_CACHE
from matplotlib2tikz import save as tikz_save

thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]

threshold_rmse = [0.9, 1.3, 1.5, 1.5, 2.2, 2.3, 2.9]

threshold_time = [462, 260, 218, 171, 139, 125, 102]

window_sizes = [25, 50, 100, 150, 200]

window_rmse = [4.0, 2.3, 1.5, 2.5, 2.0]

color1 = '#1f77b4'
color2 = '#ff7f0e'

# Threshold
plt.figure(figsize=(4 * 1.2, 2 * 1.2))
plt.plot(thresholds, threshold_rmse, marker='o')
plt.xlabel('Reliable Threshold Size')
plt.ylabel('RMSE')
plt.tight_layout()
plt.savefig(GRAPH_CACHE + 'threshold.pdf')
tikz_save(LTX_CACHE + 'threshold.tex')

# Threshold time

fig, ax = plt.subplots(figsize=(4 * 1.2, 2 * 1.2))
ax.plot(threshold_time, threshold_rmse, marker='o')
for i, txt in enumerate(thresholds):
    if txt == 2.5:
        txt = '   ' + str(txt)
    else:
        txt = ' ' + str(txt)
    ax.annotate(txt, (threshold_time[i], threshold_rmse[i]))
plt.xlabel('Max Time Between Readings (Seconds)')
plt.ylabel('RMSE')
plt.ylim(top=3.1)
plt.xlim(right=499)
plt.tight_layout()
plt.savefig(GRAPH_CACHE + 'threshold-time.pdf')
tikz_save(LTX_CACHE + 'threshold-time.tex')

# Window
plt.figure(figsize=(4 * 1.2, 2 * 1.2))
plt.plot(window_sizes, window_rmse, marker='o')
plt.xlabel('Signal Window Size')
plt.ylabel('RMSE')
plt.tight_layout()
plt.savefig(GRAPH_CACHE + 'window.pdf')
tikz_save(LTX_CACHE + 'window.tex')
