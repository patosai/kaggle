#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data/train.csv")
correlations = data.corr()
ax = sns.heatmap(
    correlations,
    xticklabels=True,
    yticklabels=True,
    cmap='viridis',
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()
