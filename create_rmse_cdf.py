import matplotlib.pyplot as plt
import numpy as np
from data import GRAPH_CACHE, LTX_CACHE
from matplotlib2tikz import save as tikz_save


maxim_rmse = [7.938651027739539, 26.517663579218432, 8.345735088481492, 5.127334300851506, 15.194416386840667, 15.491406008163652, 20.714832127599003, 14.708403628609288, 23.40337422609582, 7.669531531557288]

enhanced_rmse = [1.3183610025210128, 9.22744605719919, 3.7047517577593743, 1.457522830775298, 16.064840956090908, 3.4345755832292943, 5.590659630529937, 11.581662306553795, 7.1769796180564365, 7.004091368438626]

classifier_rmse = [0.14293587792089893, 0.9423599947537197, 1.2008502932695067, 0.9875412566076903, 1.5387876344707219, 1.3555877687997104, 1.809212912905007, 1.9721681246760638, 2.8312190297737105, 2.52124047769974]

rmses = [maxim_rmse, enhanced_rmse, classifier_rmse]

plt.figure()

for data in rmses:
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    plt.plot(sorted_data, yvals)

plt.legend(['Baseline', 'Enhanced', 'WristO2'])
plt.ylim(0.0, 1.0)
plt.xlabel('RMSE')

plt.savefig(GRAPH_CACHE + 'cdf-rmse.pdf')
tikz_save(LTX_CACHE + 'cdf-rmse.tex')
