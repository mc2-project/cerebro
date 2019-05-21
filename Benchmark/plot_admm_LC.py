#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
# import seaborn as sns
import sys
import datetime
import glob
from itertools import cycle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FixedFormatter
# %matplotlib inline

import warnings
import matplotlib.cbook
import matplotlib as mpl
import random

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath,mathptmx}']

warnings.filterwarnings('ignore')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams.update({'figure.max_open_warning': 0})
mpl.rcParams.update({"font.size": 34,"figure.autolayout": True})
mpl.rc("axes", edgecolor="0.8")
plt.rcParams["font.family"] = "Times New Roman"

legendsize = 40

red = "#a93226"
blue = "#2874a6"
green = "#1e8449"

def plot_admm():
    # x represents the dimensions of the dataset.
    x = [5, 10, 20, 30, 40]
    # Local Compute Time, no estimates
    y1 = [95.21107599, 166.3214619, 324.5535871, 504.5941834, 706.8332509]
    y2 = [3736.564337, 14038.20616, 58684.93763, 146994.9658, 296647.1091]
            
    # Log scale
    plt.yscale('log')

    plt.plot(x, y1, marker="^", linestyle="solid", markersize=8, color=red, label="Use LC")
    plt.plot(x[:-1], y2[:-1], marker='x', linestyle="solid", markersize=8, color=blue, label="Not use LC")
    plt.plot([30, 40], [146994.9658, 296647.1091], marker='x', linestyle=':', markersize=8, color=blue, label="Not use LC (estimated)")

    axes = plt.gca()
    axes.set_ylim(0, 350500)
    #axes.set_yscale('log')

    plt.legend(ncol=2, columnspacing=0.2, fontsize=15, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0)

    plt.ylabel("Time (s)",fontsize=15)
    
    plt.yticks([10, 100, 1000, 10000, 100000, 1000000], fontsize=12)
    
    plt.xlabel("\# features in the dataset", fontsize=15)
    xts = [0, 5, 10, 20, 30, 40]
    plt.xticks(xts, ["$0$", "$5$", "$10$", "$20$", "$30$", "$40$"], fontsize=12)

    fig = plt.gcf()
    fig.set_size_inches(6, 4, forward=True)

    pp = PdfPages('ADMM_Local_Compute.pdf')
    plt.savefig(pp, format='pdf', bbox_inches='tight', dpi=fig.dpi)
    pp.close()
    plt.show()
plot_admm()
