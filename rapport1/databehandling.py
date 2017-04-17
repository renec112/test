from scipy import optimize as opt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
# %% Define funcion
def grundFrekvens(F, u):
    l = 0.60; #m
    return (0.5/l)*np.sqrt(F/u)
# %% - - - - - - - - - - data - - - -  - - - - - - - -


# Dataopsamling delt op i frekvenser/lydstyrke og spændingsforskel/tid
# +- for frekvenser 0.5 Hz
#Hejsa LAUR
# Hej rallemus
# hejsa

#Snor 1: gul
# Snor 1: - 35 N +-1
# grundtone: 53.3 Hz
# 1. overtone: 108.8 Hz
# 2. overtone: 159.0 Hz
mu_gul = 8.72*1e-3
print(mu_gul)
# Snor 1: - 25 N +- 1
# grundtone: 43.9 Hz

# Snor 1: - 30 N +- 1
# grundtone: 49.4 Hz

# Snor 1: - 40 N +- 1
# grundtone: 58.0 Hz

# Snor 1: - 45 N +- 1
# grundtone: 63.1 Hz
# %% - - - - - - - - - Frekvens som funktion af snorspænding - - - - - - - - - -
data_snor1 = np.array([[43.9, 49.4, 53.3, 58.0, 63.1, 80.2], [25, 30, 35, 40, 45, 50]])# grundtoner i Hz
F_range = np.linspace(20,60,1000);
freq_F = grundFrekvens(F_range,mu_gul)
#%% ----------------- fit af Frekvens som funktion af snorspænding
xdata = data_snor1[1];
ydata = data_snor1[0];
#func er grundFrekvens()
def func(mu_fit):
    return 0.5/l*np.sqrt(xdata/mu_fit);

u_opt,u_cov = opt.curve_fit(grundFrekvens,xdata,ydata);
freq_opt = grundFrekvens(F_range,u_opt);
plt.errorbar(data_snor1[1],data_snor1[0],xerr=0,yerr=2,label = 'Observation', fmt='o',  capsize=2);
plt.plot(F_range,freq_opt,'b--',label = 'Fit');
plt.plot(F_range,freq_F, 'r--',label='Teoretisk');
plt.legend()
plt.xlabel('$F_{snor}$')
plt.ylabel('$f_g$')
plt.tight_layout();
plt.savefig("frekvensSnorSpaending.png", dpi = 199)
plt.show()
#%% - - - - - - - - - Frekvens som funktion af masse pr længde
data_30_N = np.array([[49.4, 140, 60.5, 144, 203.3],[8.72/1000, 1.00/1000, 5.79/1000, 7.87/1000, 0.564/1000]]) #grundtone, mu
mu_range = np.linspace(0.2,10,1000)/1000
freq_mu = grundFrekvens(30,mu_range)
plt.plot(mu_range,freq_mu,'r--')
plt.plot(data_30_N[1],data_30_N[0],'k.')
plt.legend(["Teoretisk værdi","Observation"])
plt.xlabel('$\mu $ [kg/m]')
plt.ylabel('$f_g $ [Hz]')
plt.tight_layout();
plt.savefig("frekvensMu.png", dpi = 199)
plt.show()
