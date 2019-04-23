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
    y1 = [119163, 119523, 113496, 106911, 102568, 100467]
    y2 = [107595, 64583.1, 42891.5, 31573.3, 24661.5, 20484.5]

    y3 = [123693.5083, 111777.5609, 108299.8904, 106640.9612, 105669.7764, 105032.0879]
    y4 = [123379.9794, 61963.79904, 41370.40651, 31050.80861, 24851.70178, 20715.89765]

    plt.plot(x, y1, linestyle=':', marker='x', markersize=7, color=red, label="SWHE")
    plt.plot(x, y2, linestyle=':', marker='o', markersize=7, color=blue, label="LHE")

    plt.plot(x, y3, linestyle='--', alpha=0.5, color='m', label="SWHE model")
    plt.plot(x, y4, linestyle='--', alpha=0.5, color=purple, label="LHE model")

    axes = plt.gca()
    axes.set_ylim([0, 145000])

    plt.legend(ncol=2, columnspacing=0.2, fontsize=15, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0)

    plt.ylabel("\# mult gates/s",fontsize=15)
    plt.yticks(np.arange(0, 145000, 20000), fontsize=12)
    plt.xlabel("\# parties in 2Gbps network", fontsize=15)
    xts = [2, 4, 6, 8, 10, 12]
    plt.xticks(xts, ["$2$", "$4$", "$6$", "$8$", "$10$", "$12$"], fontsize=12)

    fig = plt.gcf()
    fig.set_size_inches(6, 4, forward=True)

    pp = PdfPages('Exp_1a_protocols_in_diff_parties.pdf')
    plt.savefig(pp, format='pdf', bbox_inches='tight', dpi=fig.dpi)
    pp.close()
    plt.show()


if __name__ == "__main__":

    if True:
        plot_dpi_ictf()
