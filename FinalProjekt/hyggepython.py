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
          'figure.figsize'      : [8.5, 3.375],
          'legend.frameon'      : False
          }

plt.rcParams.update(params)

plt.rc('text',usetex =True)
plt.rc('font', **{'family' : "sans-serif"})

# %% on off data
def onOffTing(x):
    return np.sin(x)
x = np.linspace(0,20,1000)
y = onOffTing(x)
for i in range (0,np.size(x)):
    if y[i] <= 0:
        y[i] = 0
    else:
        y[i] = 1

# %% plot on off
plt.figure()
plt.plot(x,y,'k--')
plt.show()

# %%
# %% rise fall
xnew = x[0:250]
yrise = 1-np.exp(-xnew)
yfall = np.exp(-xnew)
plt.figure()
plt.plot(xnew,yrise,'k--')
plt.plot(xnew+5,yfall,'k--')
plt.plot(10+xnew,yrise,'k--')
plt.plot(10+xnew+5,yfall,'k--')
plt.plot(20+xnew,yrise,'k--')
plt.plot(20+xnew+5,yfall,'k--')
plt.plot(x,y,'b--')
plt.show()
