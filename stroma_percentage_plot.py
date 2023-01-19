# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:12:21 2020

@author: whitma01
"""


import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# sns.set(style="whitegrid")
sns.set_style("ticks")

colors = [(29/256.0, 236/256.0, 244/256.0),(253/256.0, 105/256.0, 179/256.0)]
cmap_name = 'my_list'
newcmp = LinearSegmentedColormap.from_list(cmap_name, colors, N=64)


data = genfromtxt('results.csv', delimiter=',')


fig = plt.figure(figsize=(3, 8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes

for n in range(1, data.shape[0]):
    print(n) 
    df=pd.DataFrame({'x': range(0,2), 'y': data[n,2:4]})
    ax.plot( 'x', 'y', data=df, marker='o', color=newcmp(n/30), linewidth=1, alpha=0.5)

print(data[1:,2])
print(data[1:,3])

data_to_plot = [data[1:,2], data[1:,3]]
positions = [0,1]
c = "black"
ax.boxplot(data_to_plot, positions=positions,
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c),
            )

# ax.set_title('lymphocyte percentages')
ax.set_xticks([0,1])
ax.set_xticklabels(['Outer','Inner'])
#plt.xlabel('Distance from tumor (um)')
plt.ylabel('Stroma percentage (%)')


r, p = stats.ttest_rel(data[1:,2], data[1:,3])
print(p)
x1, x2 = 0, 1 
y, h, col = data[1:,2].max() + 3, 1, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
plt.text((x1+x2)*.5, y+h, "p={:.3f}".format(p), ha='center', va='bottom', color=col)

plt.savefig("stroma_percentage.png", bbox_inches = 'tight')
