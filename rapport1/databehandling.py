from scipy import special, optimize
import numpy as np
import matplotlib.pyplot as plt

# %% - - - - - - - - - - data - - - -  - - - - - - - -


# Dataopsamling delt op i frekvenser/lydstyrke og spændingsforskel/tid
# +- for frekvenser 0.5 Hz

l = 0.60; #m

#Hejsa LAUR
# Hej rallemus
# hejsa

#Snor 1: gul
# Snor 1: - 35 N +-1
# grundtone: 53.3 Hz
# 1. overtone: 108.8 Hz
# 2. overtone: 159.0 Hz

# Snor 1: - 25 N +- 1
# grundtone: 43.9 Hz

# Snor 1: - 30 N +- 1
# grundtone: 49.4 Hz

# Snor 1: - 40 N +- 1
# grundtone: 58.0 Hz

# Snor 1: - 45 N +- 1
# grundtone: 63.1 Hz

data_snor1 = np.array([[43.9, 49.4, 53.3, 58.0, 63.1, 80.2], [25, 30, 35, 40, 45, 50]])# grundtoner i Hz
data_30_N = np.array([[49.4, 140, 60.5, 144, 203.3],[8.72, 1.00, 5.79, 7.87, 0.564]]) #grundtone, mu
