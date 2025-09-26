# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


data1 = np.genfromtxt("average80-86.txt")


time = np.array([])
for i in range(80000):
    time = np.append(time,i/5000.)


# リストをグラフ化
plt.plot(time,data1,"r")

plt.xlim(0, 16)
#plt.ylim(0, 1)

plt.title('Output', fontsize = 20)
plt.xlabel('Time[s]', fontsize = 16)
plt.ylabel('Output[mV]', fontsize = 16)
plt.show()







