## Preamble
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import optimize as opt
from scipy import stats
from scipy.stats import t
# t0 = t.ppf(alpha, f)
# tcdf = d.cdf(|t|m f)

# MatploLib kører TeX
params = {'legend.fontsize': '20',
          'axes.labelsize':  '20',
          'axes.titlesize':  '20',
          'xtick.labelsize': '20',
          'ytick.labelsize': '20',
          'legend.numpoints': 1,
          'text.latex.preamble' : [r'\usepackage{siunitx}',
                                   r'\usepackage{amsmath}']
          }

plt.rc('text',usetex=True)
plt.rc('font', **{'family' : "sans-serif"})
plt.rcParams.update(params)

# Faste parametre
n_brydning = 2.21      # brydningsindeks
lambda_l = 911*10**-9  # lysets bølgelængde (vakuum)
L = 3.00 * 10**-2      # gitterets længde (måling)
n = np.array([0, 1, 2, 3])

# Målte data
output = np.array([3.4]) # lydfrekvens
l = 29.8               # længde mellem AOM og pap

# Målte afstande mellem pletter
# Afstand fra 0 til 2
f_lyd = np.array([2.0, ]) * 10**8
a = np.array([2.8 ]) + np.array([0.1])


# Funktioner
# Kvalitetsparameter Q (>> 1 for Bragg og <<1 for Kaman Nalk)
def kvalitetsparameter(lambda_s, lambda_l, n, L):
    Q = 2*np.pi*lambda_l*L / (n*lambda_s**2)
    return(Q)
#Q = kvalitetsparameter(lambda_s, lambda_l, n, L)
#print(Q)

# Lydens hastighed
def v_s(f_s, lambda_s):
    v_s = lambda_s * f_s
    returN(v_s)

#v_s = v_s(f_s, lambda_s)
#print(v_s)

# Lys n'te ordens frekvens
def f_n(f, n, f_s):
    f_n = f + n*f_s
    return(f_n)

# Bølgekonstanten
def k_s(lambda_s):
    #k_s = 2*np.pi/lambda_s
    2*k*np.sin(theta_B)
    return(k_s)


# Braggs betingelse (Q>>1)
# Kun en orden
def Bragg(lambda_l, f_s, n_brydning, v_s):
    # sin(theta) approx theta
    theta_B = lambda_l * f_s / (2*n_brydning*v_s)
    theta_sep = 2*theta_B
    return(theta_B, theta_sep)

#theta_B = Bragg(lambda_l, f_s, n_brydning, v_s)

# Lydens bølgelængde
def lambda_s(lambda_l, theta_B):
    lambda_s = lambda_l / (2*np.sin(theta_B))
    return(lambda_s)
##lambda_s(theta_B)

# Effektivitet
I_0 = np.array([3.0])# dBm
def intensitet():
    p_0 = (lambda_l**2 / (2*n_xxx)) * (H/L)
    eta = (np.pi**2 / 4) * p/p_0
    I_1 = I_0*(np.sin(np.sqrt(eta)))**2
    return(I_1)

# Data
# Første modul
# Forsøg 1: Målte afstand til 1te orden (Rene, Rasmus og Laurits)
d1 = np.array([0.7, 0.8, 0.9, 1.1, 1.1, 1.2, 1.3, 1.3, 1.4, 1.5, 1.5, 1.6, 1.7,
    1.8, 1.8, 1.9, 2.0])
d2 = np.array([0.7, 0.8, 0.9, 0.9, 1.2, 1.2, 1.3, 1.4, 1.6, 1.6, 1.7, 1.8, 1.8,
    1.9, 1.9, 2.0, 2.0])
d3 = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.3, 1.3, 1.4, 1.5, 1.6, 1.6, 1.7,
    1.8, 1.9, 1.9, 2.0]) 

# De justerede frekvenser af lyd
f = np.array(np.linspace(120, 280, 17))

# Figur
plt.figure()
plt.title("Afstand til forste orden per frekvens")
plt.plot(f, d1, 'ro', label='d1')
plt.plot(f, d2, 'bo', label='d2')
plt.plot(f, d3, 'go', label='d3')
plt.ylabel("Observeret afstand")
plt.xlabel("Fast frekvens")
plt.legend()
plt.grid()

# Forsøg 2: Intensitet
data = np.array([ [3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8,
    0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4, -1.6, -1.8,
    -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0,
    3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0],

    [25.34, 24.38, 23.45, 22.21, 21.30, 20.27, 19.17, 18.32, 17.20, 16.46,
        15.51, 14.78, 13.92, 13.18, 12.62, 12.16, 11.55, 11.07, 10.52, 10.06,
        9.52, 9.09, 8.57, 8.11, 7.52, 7.34, 6.56, 5.81, 5.13, 4.58, 4.05, 3.06,
        2.94, 2.43, 2.03, 1.71, 1.46, 26.78, 27.83, 29.15, 30.02, 31.46, 32.61,
        33.81, 35.20, 36.51, 38.14]])

dBm = data[0]
P = data[1]
dBm2 = np.power(10, dBm)/10

# Figur
# Plottede regulær dBm - fejlagtigt
#plt.plot(dBm, P, 'ko', label='Plot')

plt.figure()
plt.plot(dBm2, P, 'ro', label='Plot2')
plt.legend()
plt.grid()

# Andet modul

    



plt.show()


# Noter
# Første modul
# Frekvensgeneratoren må ikke skrue op på højere end 5 dBm

# Starter med at indstille setup - rykke på krystal / pap, men spejle/linser
# var fastsat

# Problemer med at få symmetriske lyspletter om 0 punktet - Vi var tilfredse med afstanden

# Andet moduk












