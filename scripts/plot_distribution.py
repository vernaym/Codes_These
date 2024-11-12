from scipy.stats import gamma, norm
import numpy as np
import matplotlib.pyplot as plt

#x = np.arange(-3, 6, 0.01)
#k = 2
##for k in np.arange(1.5, 3, 0.5):
#theta = 1/np.sqrt(k)
#obs = 5
#sd = 5
##shift = (k-1)*theta  # mode
#shift = k*theta  # mean
#gm = gamma.pdf(x+shift, k, scale=theta)
#gs = norm.pdf(x)
##y = 10+5*gm+10*0.2*gs
#y = 10+10*0.2*gs
#plt.plot(x, y, label=k)
#print(np.mean(y))

#obs = 20
#sd1 = 2
#sd2 = 3
#k = 3
#theta = np.sqrt(1/k)  # Ensure a variance of 1 (var=k*theta^2)
#shift = k*theta  # shift = mean
#analysis = []
#for i in range(1600):
#    gamma = np.random.gamma(k, scale=theta)
#    gauss = np.random.normal(loc=0.0, scale=1.0, size=1)[0]
#    ana = obs + gauss*sd1
#    ana = ana + (gamma-shift)*0.1*obs
#    gamma = np.random.gamma(k, scale=theta)
#    ana = ana + (gamma-shift)*sd2
#
#    analysis.append(ana)
#
#mu = np.mean(analysis)
#std = np.sqrt(np.mean((analysis-mu)**2))
#x = np.linspace(10, 40, 10000)
#plt.plot(x, norm.pdf(x, loc=mu, scale=std), color='k')
#plt.bar(analysis, norm.pdf(analysis, loc=mu, scale=std), width=0.1, color='k')
#plt.bar(mu, 0.2, color='blue', width=0.1, label='ensemble mean')
#plt.bar(obs, 0.2, color='red', width=0.1, label='observation')

x = np.linspace(-6, 6, 1000)
plt.plot(x, norm.pdf(x, loc=0, scale=1), color='k')
plt.plot(x, norm.pdf(x, loc=2, scale=1), color='red')
plt.plot(x, norm.pdf(x, loc=-2, scale=1), color='blue')
plt.ylim(0, 0.6)

plt.legend()
plt.show()
