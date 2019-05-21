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

    ind = np.arange(4)
    x_labels = ['training', '+ diff privacy', '+ valid. (100)', '+ valid. (200)']

    y_1000 = [128.0712378, 128.1093448, 137.5917226, 180.4881923]
    y_5000 = [702.9440389, 702.9821459, 712.4645238, 755.3609934]
    y_10000 = [1402.588078, 1402.626185, 1412.108563, 1455.005032]
    #y_15000 = [2100.252117, 2100.290224, 2109.772602, 2152.669071]
    y_20000 = [2804.306156, 2804.344263, 2813.82664, 2856.72311]
    #y_27000 = [3797.517133, 3797.55524, 3807.037617, 3849.934087]

    # y_training = [702.944, 1402.588, 2100.252, 2804.306]
    # y_privacy = [702.982, 1402.626, 2100.290, 2804.344]
    # y_valid_100 = [712.464, 1412.109, 2109.773, 2813.827]
    # y_valid_500 = [755.361, 1455.005, 2152.669, 2856.723]

    width = 0.2

    p1 = plt.bar(ind, y_1000, width, label = 'n = 1000', alpha = 0.7, color=red, edgecolor="black")
    p2 = plt.bar(ind + width, y_5000, width, label = 'n = 5000', hatch='/', alpha = 0.7, color=blue, edgecolor="black")
    p3 = plt.bar(ind + width * 2, y_10000, width, label = 'n = 10000', hatch='+', alpha = 0.7, color=green, edgecolor = "black")
    p4 = plt.bar(ind + width * 3, y_20000, width, label = 'n = 20000', hatch='x', alpha = 0.7, color=purple, edgecolor = 'black')

    axes = plt.gca()
    axes.set_ylim([0, 4050])

    plt.legend(ncol=2, columnspacing=0.2, fontsize=15, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0)

    plt.ylabel("time (s)",fontsize=15)
    plt.yticks(np.arange(0, 4050, 800), fontsize=12)
    plt.xticks([p + 1.5 * width for p in ind], ["training", "privacy", "validate (100)", "validate (500)"], fontsize=12)

    plt.xlabel("\# training samples", fontsize = 15)

    fig = plt.gcf()
    fig.set_size_inches(6, 4, forward=True)

    pp = PdfPages('Exp_9a_policy.pdf')
    plt.savefig(pp, format='pdf', bbox_inches='tight', dpi=fig.dpi)
    pp.close()
    plt.show()


if __name__ == "__main__":

    if True:
        plot_dpi_ictf()
