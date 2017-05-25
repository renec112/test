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

# %% on off data
def onOffTing(x):
    return np.sin(x*0.65)
x = np.linspace(0,40,1000)
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
xnew = x[0:120]
yrise = 1-np.exp(-xnew)
yfall = np.exp(-xnew)

# %% BEST PLOT EVAR
plt.figure()
# plot boolean
plt.plot(x,y,'b--')
#plot 90 og 10
# plt.plot([0, 25], [0.1, 0.1],'r--')
# plt.plot([0, 25], [0.9, 0.9],'r--')
# plot intensitet
plt.plot(xnew,yrise,'k--')
plt.plot(xnew+4.7,yfall,'k--')
plt.plot(9.5+xnew,yrise,'k--')
plt.plot(9.5+xnew+5,yfall,'k--')
plt.plot(19.2+xnew,yrise,'k--')
plt.plot(19+xnew+5,yfall,'k--')
plt.plot(10+20+xnew,yrise,'k--')
plt.plot(10+19+xnew+5,yfall,'k--')
plt.plot(19.5+19+xnew[0:40],yrise[0:40],'k--')
plt.legend(["On off boolean", "Intensitet procent"],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel("t")
plt.ylabel(r"\% og boolean")
plt.axis([0,25,-0.1,1.1])
plt.savefig("tegninger/risefall.png")
plt.show()
