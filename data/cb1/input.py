# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


data1 = np.genfromtxt("wn_0.0002s.txt")


time = np.array([])
for i in range(len(data1)):
    time = np.append(time,i/5000.)


# リストをグラフ化
plt.plot(time,data1,"b")

plt.xlim(0, 16)
#plt.ylim(0, 1)

plt.title('Input', fontsize = 20)
plt.xlabel('Time[s]', fontsize = 16)
plt.ylabel('Input[pA]', fontsize = 16)
plt.show()







