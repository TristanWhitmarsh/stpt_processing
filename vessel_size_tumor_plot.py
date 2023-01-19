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

pixel_length = 0.5034
pixel_length_mm = 0.0005034
pixel_size = (pixel_length * pixel_length) # microns

# sns.set(style="whitegrid")
sns.set_style("ticks")

colors = [(29/256.0, 236/256.0, 244/256.0),(253/256.0, 105/256.0, 179/256.0)]
cmap_name = 'my_list'
newcmp = LinearSegmentedColormap.from_list(cmap_name, colors, N=64)


data = genfromtxt('vessel_size_tumor.csv', delimiter=',')
data *= pixel_size

fig = plt.figure(figsize=(9, 9))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes

for n in range(1, 29):
    ax.plot( range(0,16), data[n,1:17], marker='o', color=newcmp(n/30), linewidth=1, alpha=0.5)

print(data[1,1:])

data_to_plot = [data[1:,1], data[1:,2], data[1:,3], data[1:,4], data[1:,5], data[1:,6], data[1:,7], data[1:,8],\
                data[1:,9], data[1:,10], data[1:,11], data[1:,12], data[1:,13], data[1:,14], data[1:,15], data[1:,16]]
positions = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
c = "black"
ax.boxplot(data_to_plot, positions=positions,
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c),
            )

# ax.set_title('lymphocyte percentages')
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
ax.set_xticklabels(['0-100','100-200','200-300','300-400','400-500','500-600','600-700','700-800','700-800','700-800','700-800','700-800','700-800','700-800','700-800','700-800'])
plt.xlabel('Distance from tumor (μm)')
plt.ylabel('Vessel size (μm^2)')
plt.ylim(0, 2000)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, rotation_mode='anchor', ha="right")



#plt.show()
for n in range(1, 15):
    r, p = stats.ttest_rel(data[1:,n], data[1:,n+1])
    print(p)
    if p < 0.05:
        plt.text(n-0.6, 0, "*", fontsize=22)


plt.savefig("vessel_size_tumor.pdf", bbox_inches = 'tight')