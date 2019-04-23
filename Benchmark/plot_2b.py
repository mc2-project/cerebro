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

def plot_dpi_ictf():
    # plt.plot(nrules, nb_vals, marker='o', linestyle='-', color=red)
    # plt.plot(nrules, sb_vals, marker='o', linestyle='-', color=green2)

    x = [2, 4, 6, 8, 10, 12]
    y1 = [1321036.8, 824559.36, 584314.88, 460398.08, 377550.08, 326270.72]
    y2 = [456021.76, 198736.64, 123660.8, 90877.44, 74868.608, 62447.232]

    y3 = [23840.9, 17393.1, 15455, 15237.5, 14838.2, 14442.4]            

    plt.plot(x, y1, linestyle='--', marker='x', markersize=7, color=red, label="LHE (d=100)")
    plt.plot(x, y2, linestyle=':', marker='o', markersize=7, color=blue, label="LHE (d=10)")

    plt.plot(x, y3, linestyle='-.', marker='s', markersize=7, color=green, label="SWHE")

    axes = plt.gca()
    axes.set_ylim([-50000, 1505000])

    plt.legend(ncol=2, columnspacing=0.2, fontsize=15, bbox_to_anchor=(0,1.02,1,4), loc="lower left",
                mode="expand", borderaxespad=0)

    plt.ylabel("\# vectorized mult gates/s",fontsize=15)
    plt.yticks(np.arange(0,1505000,300000), fontsize=12)
    plt.xlabel("\# parties in 100Mbps network", fontsize=15)
    xts = [2, 4, 6, 8, 10, 12]
    plt.xticks(xts, ["$2$", "$4$", "$6$", "$8$", "$10$", "$12$"], fontsize=12)

    fig = plt.gcf()
    fig.set_size_inches(6, 4, forward=True)

    pp = PdfPages('Exp_2b_protocols_in_diff_parties_vectorsize.pdf')
    plt.savefig(pp, format='pdf', bbox_inches='tight', dpi=fig.dpi)
    pp.close()
    plt.show()


if __name__ == "__main__":

    if True:
        plot_dpi_ictf()
