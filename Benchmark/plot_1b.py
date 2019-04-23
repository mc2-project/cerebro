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

def plot_dpi_ictf():
    # plt.plot(nrules, nb_vals, marker='o', linestyle='-', color=red)
    # plt.plot(nrules, sb_vals, marker='o', linestyle='-', color=green2)

    x = [2, 4, 6, 8, 10, 12]
    y1 = [23840.9, 17393.1, 15455, 15237.5, 14838.2, 14442.4]
    y2 = [31945.8, 11774.3, 7602.14, 5661.11, 4531.43, 3761.61]

    y3 = [23883.08158, 17213.69742, 15747.82956, 15104.6939, 14743.42423, 14512.02793]
    y4 = [27973.11135, 12168.59331, 7775.510469, 5713.010194, 4515.300067, 3732.744795]
            
    plt.plot(x, y1, linestyle=':', marker='x', markersize=7, color=red, label="SWHE")
    plt.plot(x, y2, linestyle=':', marker='o', markersize=7, color=blue, label="LHE")

    plt.plot(x, y3, linestyle='--', alpha=0.5, color='m', label="SWHE model")
    plt.plot(x, y4, linestyle='--', alpha=0.5, color=purple, label="LHE model")

    axes = plt.gca()
    # axes.set_xscale("log", basex=2)
    axes.set_ylim([0, 32500])

    plt.legend(ncol=2, columnspacing=0.2, fontsize=15, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0)

    plt.ylabel("\# mult gates/s",fontsize=15)
    plt.yticks(np.arange(0,32500, 4000), fontsize=12)
    plt.xlabel("\# parties in 100Mbps network", fontsize=15)
    xts = [2, 4, 6, 8, 10, 12]
    plt.xticks(xts, ["$2$", "$4$", "$6$", "$8$", "$10$", "$12$"], fontsize=12)

    fig = plt.gcf()
    fig.set_size_inches(6, 4, forward=True)

    pp = PdfPages('Exp_1b_protocols_in_diff_parties.pdf')
    plt.savefig(pp, format='pdf', bbox_inches='tight', dpi=fig.dpi)
    pp.close()
    plt.show()


if __name__ == "__main__":

    if True:
        plot_dpi_ictf()
