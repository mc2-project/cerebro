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
purple = '#800080'

def plot_lr():
    # plt.plot(nrules, nb_vals, marker='o', linestyle='-', color=red)
    # plt.plot(nrules, sb_vals, marker='o', linestyle='-', color=green2)
    x = [1000, 5000, 10000, 15000, 20000, 27000]
    y1 = [37.95, 200.84, 398.38, 593.94, 795.89, 1093.88]
    y2 = [1209.14]
    
    plt.yscale('log')

    plt.plot(x, y1, linestyle = "solid", marker="^", markersize=8, color=purple, label="Arithmetic")
    
    plt.plot(x, [y2[0] * i//128 for i in x],  marker='x', linestyle='--', markersize=8, color=green, label="Boolean (estimated)")
    
    axes = plt.gca()
    axes.set_ylim([0, 1005000])

    plt.legend(ncol=2, columnspacing=0.2, fontsize=15, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0)

    plt.ylabel("Time (s)",fontsize=15)
    plt.yticks([10, 100, 1000, 10000, 100000, 1000000], fontsize=12)
    
    plt.xlabel("Number of samples", fontsize=15)
    xts = [0, 6000, 12000, 18000, 24000, 30000]

    plt.xticks(xts, ["$0$", "$6000$", "$12000$", "$18000$", "$24000$", "$30000$"], fontsize=12)

    fig = plt.gcf()
    fig.set_size_inches(6, 4, forward=True)

    pp = PdfPages('LR.pdf')
    plt.savefig(pp, format='pdf', bbox_inches='tight', dpi=fig.dpi)
    pp.close()
    plt.show()
plot_lr()
