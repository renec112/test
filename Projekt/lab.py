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





# %% Faste parametre
n_brydning = 2.21                   # brydningsindeks
lambda_l   = 911*10**-9             # lysets boelgelaengde (vakuum)
L          = 3.00 * 10**-2          # gitterets laengde (maaling)
n          = np.array([0, 1, 2, 3]) # Observarbare ordner








# - - - - - - - - - - - - - MODUL 1 - - - - - - - - - - - - - -
# Når vi ikke skruede på noget så stod: dBm = 3.0


# Forsoeg 1: Maalte afstand til 1te orden (Rene, Rasmus og Laurits)
d1 = np.array([0.7, 0.8, 0.9, 1.1, 1.1, 1.2, 1.3, 1.3, 1.4, 1.5, 1.5, 1.6, 1.7,
    1.8, 1.8, 1.9, 2.0]) * 10**-2
d2 = np.array([0.7, 0.8, 0.9, 0.9, 1.2, 1.2, 1.3, 1.4, 1.6, 1.6, 1.7, 1.8, 1.8,
    1.9, 1.9, 2.0, 2.0]) * 10**-2
d3 = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.3, 1.3, 1.4, 1.5, 1.6, 1.6, 1.7,
    1.8, 1.9, 1.9, 2.0])  * 10**-2

# Definerer en gennemsnitlig afstand
d4 = np.array([d1, d2, d3]).T
d  = (d1 + d2 + d3)/3

theta_sep = d / l # Skøn over theta_sep -- array


# Nu skal vi finde usikkerhed på d, derfor defineres en masse 0'er.
# Bagefter looper vi over d1,d2,d3 sådan at når vi tager standardafvigelsen så
# får vi standardafvigelsen for de tre i'te indgange i d1,d2,d3.
sds_d = np.zeros(np.size(d1)) #

for i in range(0, np.size(d1)):
    sds_d[i] = np.std(d4[i])
# %%
#Nu har vi standardafvigelse men der er ogsaa en usikkerhed i at bruge en lineal
#altsaa skal sds_d_samlet vaere de to sammenlagt
sds_maaling = 1./1000             # usikkerhed paa maaling 1 mm pga. lineal
sds_d       = sds_d + sds_maaling #SAMLET usikkerhed på d


# frekvenser for Frekvensgeneratoren i forsøget hvor vi varierede disse
fs     = np.array(np.linspace(120, 280, 17)) * 10**6 # går fra 120 MHz til 280 MHz
sds_fs = 0 # Indtil videre - spoerg Andreas

# usikkerhed på theta_sep, fundet vha. ophobningsloven.
sds_theta_sep = np.sqrt((1/l**2) * sds_d**2 + (d/l**2)**2 * sds_l**2)
# usikkerhed på lydhastighed, fundet vha. ophobningsloven, bemærk at det første led er 0 da sds_fs = 0
sds_vs        = np.sqrt((lambda_l/theta_sep)**2 * sds_fs**2 + (lambda_l*fs /
    theta_sep**2)**2 * sds_theta_sep**2)
# - - - - - - - - - - - - - FIT hvor vi vil have lydhastighed - - - - - - - -
def thetaFit(fs, k,c):
    theta_sep = k*fs+c  # k = lambda/v_s, c er et konstant led for at ramme data bedre
    return(theta_sep)
#fitter data ved funktion thetaFit, med xdata fs og ydata theta_sep
p_opt, p_cov = opt.curve_fit(thetaFit, fs, theta_sep)
k            = p_opt[0] # 0'te indgang i p_opt pga. funktionen
c            = p_opt[1] # 1'te indgang i p_opt pga. funktionen
v_s          = lambda_l/k # lydhastighed fundet vha. fit ovenfor

x_lin        = np.linspace(-0.2*10**8, fs[-1]+2*10**8, 100) # en linspace for at kunne plotte den fittede kurve med theta som funktion af fs
theta_fit    = thetaFit(x_lin,k,c)



# $$ - - - - - - - - - - - - - Plot - - - - - - - - - - - - - -
fs_plt = fs/10**6
x_lin_plt = x_lin/10**6

limits_dplt  = [-10, 295, -0.005, 0.075] #graenser til plot nedenfor

plt.figure()
plt.title(r"Plot af $\theta_{sep}$ som funktion af $f_s$")
plt.errorbar(fs_plt,theta_sep,fmt          = 'ko', xerr = sds_fs, yerr = sds_theta_sep)

plt.plot(x_lin_plt,theta_fit, '--b', label ="fit")
plt.ylabel(r"$\theta_{sep} \ \left[ \si{\radian}\right]$")
plt.xlabel(r"$f_s \ \left[ \si{\mega\hertz}\right]$")
plt.legend(['Fit','Datapunkter'],loc   = 2)

plt.axis(limits_dplt)
# plt.savefig('tegninger/graf1.png')

# %%
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

# Krystallens bredde og hoejde (mm)
B = 0.5 * 10**(-3)
L = 2.00 * 10**(-3)

dBm     = data[0] #power i vibrator
P       = data[1] #intensitet i uW
dBm2    = np.power(10, dBm/10.)
sds_P   = 3 #TODO find ud af hvad usikkerhed er på denne måling
sds_dbm = 0


# - - - - - - - - - - - - - plot - - - - - - - - - - - - - -
P_sorted   = np.sort(P)
dBm_sorted = np.sort(dBm2)
plt.figure()
plt.plot(dBm_sorted,P_sorted,'rx')
special.errorfill(dBm_sorted,P_sorted,sds_P,alpha_fill = 0.2,color = 'red')
plt.xlabel(r'$P \left[ \si{\milli\watt}\right]$')
plt.ylabel(r'$I_1 \ \left[ \si{\micro\watt}\right]$')
plt.legend(['Datapunkter' ,'Linje genne punkter','Usikkerheder'])
plt.grid()
# plt.savefig('tegninger/graf2.png')

# Andet modul

# Noter
# Foerste modul
# Frekvensgeneratoren maa ikke skrue op paa hoejere end 5 dBm

# Starter med at indstille setup - rykke paa krystal / pap, men spejle/linser
# var fastsat

# Problemer med at faa symmetriske lyspletter om 0 punktet - Vi var tilfredse med afstanden

# Andet moduk







# Noter:




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



maxi   = 417
mini   = 10
deltay = maxi - mini




# Fit
def Errorfunction(x, w0):
    indmad = (x*np.sqrt(2)/w0)
    y = erf(indmad)
    return(y)

p_opt, p_cov = opt.curve_fit(Errorfunction, x, Intensitet)#, bounds=(0.05, 0.50))
w0 = p_opt


estimat = Intensitet*Errorfunction(x, w0)
# %% plot
plt.figure()
plt.grid()
plt.plot(x, Intensitet, 'ro', label='Data')
plt.plot(x, estimat, 'b-', label='Fit')
plt.xlabel(r'$x \ \left[\si{\milli\meter}\right]$')
plt.ylabel(r'$I \ \left[\si{\micro\watt}\right]$')
plt.legend()
# plt.savefig('tegninger/graf3.png')







# %% w0
Imax = 435 # mikrowatt
I    = np.array([0.84*Imax, 0.16*Imax])
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

i  = np.ones(np.size(risetime))* w0# * 0.2 * 10**-3

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

plt.show()
