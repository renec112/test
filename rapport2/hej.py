# %%  - - - - - - - - Import  - - - - - - - -
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
plt.rcParams['legend.numpoints']=1

#positioner paa ting
I_0 = 15/100. #position for image 0
y_0 = 2.5/100. # hoejden af image 0

#usikkerheder
pm_y = 0.1/100 # usikkerheder paa hoejde
pm_x = 1/100# usikkerhed paa hvor fokus er. valgt til at vaere konstant 1 cm.

#linsers orden [+10],[+5]
I_r1 = np.array([58.,64.,59.,57.5,57.,57.8,59.8,61.,62.6,63.9,66.])/100. # position for reelt image paa papir
I_r2 = np.array([36.,36.6,37.9,39.2])/100.                               # En kommentar
lin1 = np.array([40,30,32,34,36,38,42,44,46,48,50])/100.                  # linse
lin2 = np.array([25,27,29,31])/100.                                      # Anden kommentar

#hoejde
y_r = np.array([1.8,5.6,4.0,3.1,2.5,2.1,1.6,1.4,1.3,1.1,1.1])/100. # hoejden af image paa papir
y_r2 = np.array([2.9,1.9,1.5,1.3])/100.                            # Kommenterer lidt mere

#laengder
s1 = lin1-I_0       # m - laengden s
s_mark1 = I_r1-lin1 # m - laengden s'
s2 = lin2-I_0       # m - laengden s
s_mark2 = I_r2-lin2 # m - laengden s'

# %% - - - - - - - - teoretisk fokuspunkter - - - - - - - -
def focal(s,s_mark):
    return (s*s_mark)/(s+s_mark)
focal_teo1 = focal(s1,s_mark1)
#print(focal_teo1)
# %% - - - - - - - - plot - - - - - - - -
s_inv1 = 1/s1
s_markinv1 = 1/s_mark1
s_inv2 = 1/s2
s_markinv2 = 1/s_mark2
<<<<<<< HEAD

=======
# figur med det raa data
>>>>>>> 0ad5f5e03ab46de02464062268eb928de843a214
plt.figure()
plt.title('Maaledata')
plt.plot(s1,s_mark1, 'bo', label="Datasaet 1")
plt.plot(s2,s_mark2, 'ro', label="Datasaet 2")
plt.legend()
plt.xlabel("s")
plt.ylabel("s'")
plt.axis([0.08, 0.4, 0.05, 0.36])


# %% - - - - - - - - fit - - - - - - - -
def func_fit(s_inv,f):
    return 1/f - s_inv
<<<<<<< HEAD
=======

>>>>>>> 0ad5f5e03ab46de02464062268eb928de843a214
f_opt1,f_cov1 = opt.curve_fit(func_fit,s_inv1,s_markinv1)
f_opt2,f_cov2 = opt.curve_fit(func_fit,s_inv2,s_markinv2)
print(f_opt1)
print(f_opt2)
range = np.linspace(2,7,1000.)
s_mark_fit1 = func_fit(range,f_opt1)
<<<<<<< HEAD
s_mark_fit2 = func_fit(range,f_opt2)
#figur for datasæt 1
plt.figure()
plt.plot(range,s_mark_fit1)
=======

#figur for datasaet 1
plt.figure()
plt.plot(range,s_mark_fit1, '--k', label='fit')
>>>>>>> 0ad5f5e03ab46de02464062268eb928de843a214
plt.plot(s_inv1,s_markinv1,'ok',label="Datasaet 1")
plt.legend()
plt.xlabel("1/s")
plt.ylabel("1/s'")
<<<<<<< HEAD
plt.show()
#figur for datasæt 2
plt.figure()

plt.plot(range,s_mark_fit2)
plt.plot(s_inv2,s_markinv2,'ok',label="Datasaet 2")
plt.legend()
plt.plot(range,s_mark_fit)
plt.plot(s_inv1,s_markinv1,'ok')
=======
plt.savefig(1.png)
#figur for datasaet 2
range2 = np.linspace(5.5,11,1000)
s_mark_fit2 = func_fit(range2,f_opt2)
plt.figure()

plt.plot(range2,s_mark_fit2, '--k', label='fit')
plt.plot(s_inv2,s_markinv2,'ok',label="Datasaet 2")
plt.legend()
>>>>>>> 0ad5f5e03ab46de02464062268eb928de843a214
plt.grid()
plt.xlabel("1/s")
plt.ylabel("1/s'")
plt.savefig("2.png")
plt.show()

