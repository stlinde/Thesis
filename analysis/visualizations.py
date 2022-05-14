# analysis/visualizations.py
"""
This module will implement functions for creating the visualizations used.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def tuftefy(ax):
    """Remove spines and tick position markers to reduce ink."""
    # 
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color('grey')

    ax.grid(color="w", alpha=0.5)
    ax.get_yaxis().grid(True)
    ax.get_xaxis().grid(False)



