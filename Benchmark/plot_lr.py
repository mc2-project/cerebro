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

def plot_lr():
    # plt.plot(nrules, nb_vals, marker='o', linestyle='-', color=red)
    # plt.plot(nrules, sb_vals, marker='o', linestyle='-', color=green2)
    x = [1000, 5000, 10000, 15000, 20000, 27000]
    y1 = [37.95, 200.84, 398.38, 593.94, 795.89, 1093.88]
    y2 = [1209.14]
    
    
    # Local Compute Time, no estimates
    #y1 = [90.93, 153.59, 281.52, 413.39, 549.59]
    #y2 = [891, 3425.86, 15675.83, 42925.32, 91685.354]
    #y1 = [133405, 127686, 125127, 119390, 113552, 111843]
    

    plt.plot(x, y1, marker="^", markersize=8, color=red, label="Arithmetic", linewidth=3)
    
    
    #plt.plot(x[0], y2[0], marker='x', markersize=8, color=blue, label="Boolean", linewidth=3)
    
    plt.plot(x, [y2[0] * i//128 for i in x], linewidth=3, marker='x', linestyle=':', markersize=8, color=blue, label="Boolean Estimated")
    

    #plt.plot(x, y3, linestyle='--', alpha=0.5, color='m', label="SH-SWHE model")
    #plt.plot(x, y4, linestyle='--', alpha=0.5, color='g', label="SH-LHE model")

    axes = plt.gca()
    axes.set_ylim([0, 300000])

    plt.legend(ncol=2, columnspacing=0.2, fontsize=15, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0)

    plt.ylabel("Time (s)",fontsize=15)
    plt.yticks(np.arange(0, 300000, 50000), fontsize=12)
    plt.xlabel("Number of samples", fontsize=15)
    xts = [0, 1000, 5000, 10000, 15000, 20000, 27000]

    plt.xticks(xts, ["0$", "$1000$", "$5000$", "$10000$", "$15000$", "$20000$", "$27000$"], fontsize=12)

    fig = plt.gcf()
    fig.set_size_inches(6, 4, forward=True)

    pp = PdfPages('LR.pdf')
    plt.savefig(pp, format='pdf', bbox_inches='tight', dpi=fig.dpi)
    pp.close()
    plt.show()
plot_lr()