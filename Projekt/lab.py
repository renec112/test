## Preamble
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

# MatploLib koerer TeX
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
n_brydning = 2.21            # brydningsindeks
lambda_l   = 911*10**-9      # lysets boelgelaengde (vakuum)
L          = 3.00 * 10**-2   # gitterets laengde (maaling)
n          = np.array([0, 1, 2, 3]) # Observarbare ordner

# Maalte data
output = np.array([3.4]) # lydfrekvens
l      = 29.8 * 10**-2            # laengde mellem AOM og pap
sds_l  = 0.1 * 10**-2

# Maalte afstande mellem pletter
# Afstand fra 0 til 2
f_lyd = np.array([2.0, ]) * 10**8
a = np.array([2.8 ]) + np.array([0.1])

# Funktioner
# Kvalitetsparameter Q (>> 1 for Bragg og <<1 for Kaman Nalk)
def kvalitetsparameter(lambda_s, lambda_l, n, L):
    Q = 2*np.pi*lambda_l*L / (n*lambda_s**2)
    return(Q)
#Q = kvalitetsparameter(lambda_s, lambda_l, n, L)
#

# Lydens hastighed
def v_s(f_s, lambda_s):
    v_s = lambda_s * f_s
    return(v_s)

#v_s = v_s(f_s, lambda_s)
#

# Lys n'te ordens frekvens
def f_n(f, n, f_s):
    f_n = f + n*f_s
    return(f_n)

# Boelgekonstanten
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

# Lydens boelgelaengde
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
# Foerste modul
# Forsoeg 1: Maalte afstand til 1te orden (Rene, Rasmus og Laurits)
d1 = np.array([0.7, 0.8, 0.9, 1.1, 1.1, 1.2, 1.3, 1.3, 1.4, 1.5, 1.5, 1.6, 1.7,
    1.8, 1.8, 1.9, 2.0]) * 10**-2
d2 = np.array([0.7, 0.8, 0.9, 0.9, 1.2, 1.2, 1.3, 1.4, 1.6, 1.6, 1.7, 1.8, 1.8,
    1.9, 1.9, 2.0, 2.0]) * 10**-2
d3 = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.3, 1.3, 1.4, 1.5, 1.6, 1.6, 1.7,
    1.8, 1.9, 1.9, 2.0])  * 10**-2

d4 = np.array([d1, d2, d3]).T
d = (d1 + d2 + d3)/3

theta_sep = d / l

sds_d = np.zeros(np.size(d1))
for i in range(0, np.size(d1)):
    sds_d[i] = np.std(d4[i])
#%%
#Nu har vi standardafvigelse men der er ogsaa en usikkerhed i at bruge en lineal
#altsaa skal sds_d_samlet vaere de to sammenlagt
sds_maaling = 1./1000 # usikkerhed paa maaling 1 mm
sds_d = sds_d + sds_maaling
# De justerede frekvenser af lyd
fs = np.array(np.linspace(120, 280, 17)) * 10**6
sds_fs = 0 # Indtil videre - spoerg Andreas

sds_theta_sep = np.sqrt((1/l**2) * sds_d**2 + (d/l**2)**2 * sds_l**2)

sds_vs = np.sqrt((lambda_l/theta_sep)**2 * sds_fs**2 + (lambda_l*fs /
    theta_sep**2)**2 * sds_theta_sep**2)

def thetaFit(fs, k):
    theta_sep = k*fs
    return(theta_sep)

p_opt, p_cov = opt.curve_fit(thetaFit, fs, theta_sep)



limits_dplt = [fs[0]-0.2*10**8,fs[-1]+0.2*10**8,d[0]-0.002,d[-1]+0.002] #graenser til plot nedenfor

print(np.size(fs),np.size(d))
# %% fit

farve = 'red'
alpha_fill = 0.2

plt.figure()
plt.title("Usikkerhedsplot med gennemsnitlig d")
plt.plot(fs,d,'ko')

plt.plot(x_lin,theta_fit, '--b', label="fit")
# special.errorfill(fs, d, sds_d,alpha_fill=alpha_fill,color=farve)
plt.ylabel("Observeret afstand")
plt.xlabel("Fast frekvens")
plt.legend(['Datapunkter','Standardafvigelse'],loc = 2)
plt.axis(limits_dplt)

# plt.grid()


# Forsoeg 2: Intensitet
data = np.array([ [3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8,
    0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4, -1.6, -1.8,
    -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0,
    3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0],

    [25.34, 24.38, 23.45, 22.21, 21.30, 20.27, 19.17, 18.32, 17.20, 16.46,
        15.51, 14.78, 13.92, 13.18, 12.62, 12.16, 11.55, 11.07, 10.52, 10.06,
        9.52, 9.09, 8.57, 8.11, 7.52, 7.34, 6.56, 5.81, 5.13, 4.58, 4.05, 3.06,
        2.94, 2.43, 2.03, 1.71, 1.46, 26.78, 27.83, 29.15, 30.02, 31.46, 32.61,
        33.81, 35.20, 36.51, 38.14]])

dBm = data[0] #power i vibrator
P = data[1] #intensitet i uW
dBm2 = np.power(10, dBm)/10
sds_P = 5 #TODO find ud af hvad usikkerhed er på denne måling
sds_dbm = 0
P_sorted = np.sort(P)
dBm_sorted = np.sort(dBm)


plt.figure()
plt.plot(dBm_sorted,P_sorted,'rx',label='Datapunkter')
special.errorfill(dBm_sorted,P_sorted,sds_P,alpha_fill = alpha_fill,color = farve)
plt.legend()
plt.grid()

# Andet modul

#plt.show()


# Noter
# Foerste modul
# Frekvensgeneratoren maa ikke skrue op paa hoejere end 5 dBm

# Starter med at indstille setup - rykke paa krystal / pap, men spejle/linser
# var fastsat

# Problemer med at faa symmetriske lyspletter om 0 punktet - Vi var tilfredse med afstanden

# Andet moduk







# Noter:

# Krystallens bredde og hoejde (mm)
B = 0.5
L = 2.00


Intensitet  = np.array([502.9, 502.1, 500.3, 496.4, 491.1, 487.3, 478.0, 471.3,
    460.8, 444.7, 428.6, 406.2, 382.5, 354.2, 322.1, 290.3, 251.7, 218.1,
    187.3, 152.4, 123.5, 101.0, 76.51, 57.61, 45.21, 31.18, 24.81, 17.28,
    14.03, 10.02, 6.982, 5.421])
    #])# mikrowatt



a = np.size(Intensitet) * 0.1
x = np.arange(3.20, 3.20 + a, 0.1) # målinger med kniv foran stråle





# MAX INTENSITET
#495
# 84 = 3.405 # mm
# 16 = 3.618

# 84 = 4.585
# 16 = 4.705

d = 4.705 - 4.585
k = 3.618 - 3.405



maxi = 417
mini = 10
deltay = maxi - mini




# Fit
def Errorfunction(x, w0):
    indmad = (x*np.sqrt(2)/w0)
    y = erf(indmad)
    return(y)

p_opt, p_cov = opt.curve_fit(Errorfunction, x, Intensitet)#, bounds=(0.05, 0.50))
w0 = p_opt


estimat = Intensitet*Errorfunction(x, w0)

plt.figure()
plt.grid()
plt.plot(x, Intensitet, 'ro', label='Data')
plt.plot(x, estimat, 'b-', label='Fit')
plt.legend()






# w0
Imax = 435 # mikrowatt
I = np.array([0.84*Imax, 0.16*Imax])
kniv = np.array([3.51, 3.60 ])*10**-3
#
#
#Imin =
#Hoej =

w0 = kniv[1] - kniv[0]

def lydhastighed(risetime, w0):
    vs = 0.64*(2*w0/ risetime)
    return(vs)

risetime = np.array([112, 200]) * 10 **-9

i = np.ones(np.size(risetime))* w0# * 0.2 * 10**-3

vs = lydhastighed(risetime, i)



# 50 OHM VED OSCILLOSKOP
# 16 OG 84 til w0
# 90 OG 10 til risetime


# Eksperimentel opstilling
EkspOps = """
Signal generator , switch, forstaerker, power kobler, AOM, men lille bid af
signal skal ud og over i oscilloskop, ogsaa daempet paa 50 dB (paa den sikre side).
11.5 db power kobling.

forbundet til 50 ohm, for at se det gule signal. tidsskala er hurtig, graen er
langsom, aom giver 200 mhz, groen (langsom) .

Groenne kanal - detektor,"""



# print(v_s)
# print(p_opt)
# print((1/p_opt) * lambda_l)
# print(Intensitet)
# print(x)
# print(np.size(x))
# print(np.size(Intensitet))
# print(d)
# print(k)
# print(deltay*0.9-mini)
# print(deltay*0.1+mini)
# print(w0)
# print(0.84*Imax)
# print(0.16*Imax)
# print(vs)
# print(EkspOps)



plt.show()
