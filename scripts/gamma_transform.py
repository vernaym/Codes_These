import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc, gammaincc, gamma

x = np.arange(0.0001, 1, 0.001)

#plt.plot(x, np.exp(-(x/0.1)**2))
#plt.show()
#sys.exit()

for k in [5, 10, 20]:
    plt.plot(x, 1-np.exp(-k*x**2), label=f'k={k}')


#for alpha in [2, 3, 4, 5]:
#for alpha in [2]:
#    for beta in [10]:
#        plt.plot(x, gammaincc(beta*x, alpha), label=f'alpha={alpha}, beta={beta}')
#plt.plot(x, gammainc(1/x, 1)/gamma(1), label=f'Gamma={gamma}')
#for gamma in [2, 3, 4]:
    #plt.plot(x, 5*x**gamma, label=f'Gamma={gamma}')
plt.ylim([0,1])
plt.xlim([0,1])
plt.legend()
plt.show()

