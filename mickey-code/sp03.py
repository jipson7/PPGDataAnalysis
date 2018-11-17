import csv
from statistics import mean, stdev
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

training = False

def cdf(data,xlabel,figure_text=None,**kwargs):

    data_size=len(data)

    # Set bins edges
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts=counts.astype(float)/data_size

    # Find the cdf
    cdf = np.cumsum(counts)

    # Plot the cdf
    plt.plot(bin_edges[0:-1], cdf, **kwargs)
    plt.ylim((0,1))
    plt.ylabel("CDF")
    plt.xlabel(xlabel)
    plt.grid(True)
    if figure_text:
        plt.text(1, .025, figure_text)


reflective = []
transitive = []



with open('csv-fingertip.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if len(row) > 0 and row[1] != "NaN" and row[1] != "nan" and row[0] != "nan":
                reflective.append(float(row[0]))
                transitive.append(float(row[1]))


print("Transitive max {} min {} mean {} stdev {}".format(max(transitive), min(transitive),mean(transitive),stdev(transitive)))
print("Reflective max {} min {} mean {} stdev {}".format(max(reflective), min(reflective),mean(reflective),stdev(reflective)))


caliber = range(100,300,1)   #[1.9,1.925,1.95,2]

lowest_error = max(transitive)

#training
if training:
    for adj in caliber:
        error = []
    
        adj = adj / 100
    
        for i in range(int(len(transitive)/2)):
                error.append(abs(round((reflective[i]-adj))-transitive[i]))
    
        print("Training adj {} mean {} stdev {}".format(adj, mean(error), stdev(error)))
    
        if (mean(error) < lowest_error):
            lowest_error = mean(error)
            optimal_adjustment = adj

    print('Optimal adjustement:', optimal_adjustment)


#testing
optimal_adjustment = 1.46
error = []
unadjusted = []

for i in range(int(len(transitive)/2),len(transitive)):
    error.append(abs(round((reflective[i]-optimal_adjustment))-transitive[i]))
    unadjusted.append(abs(round((reflective[i]))-transitive[i]))


print('Optimal adjustement', optimal_adjustment)
print("Testing adj {} mean {} stdev {}".format(optimal_adjustment, mean(error), stdev(error)))


figure_text = r"$\mu={:.3} \ \sigma={:.3}$".format(mean(error), stdev(error))
print(figure_text)

plt.figure(figsize=(6.2*1.2,2*1.2))
cdf(unadjusted,"Absolute difference", ls='--', marker='o',label='no calibration')
cdf(error,"Absolute difference", marker='s',color='C1', label='recalibrated')
plt.ylim(top=1.1)
plt.legend(handlelength=3)
plt.tight_layout()
plt.savefig('fingertip-calibration.pdf')

plt.show()
