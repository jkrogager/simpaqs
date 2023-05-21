# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.table import Table


plt.close()

def line(x, a, b):
    return a + b*x

data = Table.read('Al32_logNHI-data.csv', comment='#')

x = data['logNHI']
y = data['Al32']
x_err = data['logNHI_err']
y_err = data['Al32_err']
plt.errorbar(x, y, xerr=x_err, yerr=y_err, marker='s', color='k', ls='')

x_norm = x - np.mean(x)
# popt, pcov = curve_fit(line, x_norm, y, [-0.81, -1.0], sigma=y_err, absolute_sigma=True)
popt, pcov = curve_fit(line, x_norm, y, [-0.81, -1.0])
scatter = np.std(y - line(x_norm, *popt))

# plotting:
X = np.linspace(x.min(), x.max(), 100)
X_norm = X - x.mean()
plt.plot(X, line(X_norm, *popt), 'r--')
plt.plot(X, line(X_norm, *popt)+scatter, 'r:')
plt.plot(X, line(X_norm, *popt)-scatter, 'r:')

perr = np.sqrt(np.diag(pcov))
print(f"a = {popt[0]:.3f} ± {perr[0]:.3f}")
print(f"b = {popt[1]:.3f} ± {perr[1]:.3f}")
print(f"scatter = {scatter:.3f}")


