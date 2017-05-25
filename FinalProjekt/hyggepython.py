# %% Import
import numpy as np
import matplotlib.pyplot as plt
from mpltools import special
from scipy.special import erf
from scipy.special import erfc
from scipy.optimize import curve_fit
from scipy import optimize as opt
from scipy import stats
from scipy.stats import t
# t0 = t.ppf(alpha, f)
# tcdf = d.cdf(|t|m f)
#



# MatploLib koerer TeX
params = {'legend.fontsize'     : '20',
          'axes.labelsize'      : '20',
          'axes.titlesize'      : '20',
          'xtick.labelsize'     : '20',
          'ytick.labelsize'     : '20',
          'legend.numpoints'    : 1,
          'text.latex.preamble' : [r'\usepackage{siunitx}',
                                   r'\usepackage{amsmath}'],
          'axes.spines.right'   : False,
          'axes.spines.top'     : False,
          'figure.figsize'      : [8.5, 6.375],
          'legend.frameon'      : False
          }

plt.rcParams.update(params)

plt.rc('text',usetex =True)
plt.rc('font', **{'family' : "sans-serif"})
