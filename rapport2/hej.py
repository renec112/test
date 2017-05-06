# Preamble
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp



# positioner på ting
I_0 = 15/100  # position for image 0
y_0 = 2.5/100 # hoejden af image 0

# position
# linsers orden [+10],[+5]
I_r1 = np.array([58.,64.,59.,57.5,57.,57.8,59.8,61.,62.6,63.9,66.])/100. # position for reelt image paa papir
I_r2 = np.array([36.,36.6,37.9,39.2])/100.
lin1 = np.array([40,30,32,34,36,38,42,44,46,48,50])/100 # linse
lin2 = np.array([25,27,29,31])/100.

# hoejde
y_r = np.array([1.8,5.6,4.0,3.1,2.5,2.1,1.6,1.4,1.3,1.1,1.1])/100. # hoejden af image paa papir
y_r2 = np.array([2.9,1.9,1.5,1.3])/100.

# leangder
s = lin-I_0 #m - laengden s
s_mark = I_r-lin #m - laengden s' 

# usikkerheder
pm_y = 0.1/100 # usikkerheder paa hoejde
pm_x = 1/100# usikkerhed paa hvor fokus er. valgt til at vaere konstant 1 cm.

# papiret bliver presset ind under maaling
plt.figure()
plt.plot()
print(np.size(lin))


