import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.interpolate import UnivariateSpline as spline

from astropy.io import fits
from astropy.table import Table

arms = ['BLUE', 'GREEN', 'RED']

resolution = {}
for arm in arms:
    l, R = np.loadtxt('LR_%s.data' % arm.lower(), unpack=True)
    fname = f'LR_resolution_{arm}.fits'
    with fits.open(fname) as hdu:
        tab = hdu[1].data
        wl = tab['WAVE']
    pixsize = 0.1
    R_interp = spline(l, R, s=0.)
    res_pixel = R_interp(wl) / wl / pixsize / 2.355
    resolution[arm] = (wl, res_pixel)

# -- Create Joint Wavelength Grid:
wl = np.linspace(3700, 9500, 201)

wd_max = np.max([np.max(resol) for _, resol in resolution.values()])
nbins = len(wl)
ndiag = int(6*np.ceil(wd_max)+1)

reso = list()

for l, wdisp in resolution.values():
    wd = np.interp(wl, l, wdisp, left=np.nan, right=np.nan)
    r = np.ones([ndiag, nbins])*np.nan
    y0 = ndiag//2
    y = np.arange(ndiag) - y0
    for i, sig in enumerate(wd):
        if sig != np.nan:
            lsf = np.exp(-0.5 * y**2 / sig**2)
            r[:, i] = lsf
    reso.append(r)
reso = np.nanmean(reso, axis=0)
reso /= np.sum(reso, axis=0)

LR = np.vstack([wl, reso])
np.savetxt('4MOST_LR_kernel.txt', LR)

