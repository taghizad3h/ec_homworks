import matplotlib.pyplot as plt
import numpy as np


path = 'second_model_cov_rosen_1000dim'
title = 'fitness of es for rosen function with dim 1000 using variacnes only'
fits = np.loadtxt(path)
x = list(range(fits.shape[0]))

plt.figure(dpi=200)
plt.plot(x, fits)
plt.xlabel('generations')
plt.ylabel('fitnes')
plt.title(title)
plt.savefig(path+'.png')
plt.show()

