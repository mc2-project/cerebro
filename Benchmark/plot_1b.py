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
    y1 = [27646, 18774.4, 17511.2, 16692.1, 16610, 16091]
    y2 = [36402.5, 13345.4, 8485.42, 6442.36, 5136.55, 4279.01]

    y3 = [27271.22545, 19234.0358, 17513.54494, 16763.78202, 16343.96639, 16075.57908]
    y4 = [31257.1195, 13738.89109, 8804.40999, 6477.824613, 5123.838602, 4238.014605]

    plt.plot(x, y1, linestyle=':', marker='x', markersize=6, color=red, label="SH-SWHE")
    plt.plot(x, y2, linestyle=':', marker='x', markersize=6, color=blue, label="SH-LHE")

    plt.plot(x, y3, linestyle='--', alpha=0.5, color='m', label="SH-SWHE model")
    plt.plot(x, y4, linestyle='--', alpha=0.5, color='g', label="SH-LHE model")

    axes = plt.gca()
    # axes.set_xscale("log", basex=2)
    axes.set_ylim([0, 40500])

    plt.legend(ncol=2, columnspacing=0.2, fontsize=15, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0)

    plt.ylabel("\# mult gates/s",fontsize=15)
    plt.yticks(np.arange(0,40500,5000), fontsize=12)
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
