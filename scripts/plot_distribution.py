from scipy.stats import gamma, norm
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3, 6, 0.01)
k = 2
#for k in np.arange(1.5, 3, 0.5):
theta = 1/np.sqrt(k)
obs = 5
sd = 5
shift = (k-1)*theta  # mode
#shift = k*theta  # mean
gm = gamma.pdf(x+shift, k, scale=theta)
gs = norm.pdf(x)
plt.plot(x, gm+gs, label=k)

plt.legend()
plt.show()
