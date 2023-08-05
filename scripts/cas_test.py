#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 31/03/2023

import os, sys
import numpy as np
import pandas as pd

import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, gamma
from scipy import signal
import random
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.sparse import csc_matrix, csr_matrix, dia_matrix, diags

from These.radar import Preprocessing_ANTILOPE

savedir = '/home/vernaym/These/figures/illustration'

Np = 15  # Domain size

def diagonal_field():
    """Generation of an idealised precipitation field"""
    # Generation of a fake ANTILOPE precipitation field
    #field = diags([1, 3, 5, 7, 10, 10, 10, 7, 5, 3, 1], [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,], shape=(Np, Np)).toarray()
    #field[3,3] = 2  # Underestimation of the precipitation in the center of the domain (==ridge)
    field = diags([10-abs(x) for x in range(-Np//2, Np//2, 1)], range(-Np//2, Np//2, 1), shape=(Np, Np)).toarray()
    field[Np//2, Np//2] = 2  # Underestimation of the precipitation in the center of the domain (==ridge)

    return field

def error_field():
    ratio =  np.ones((Np, Np))
    ratio[(Np-1)//2, (Np-1)//2] = 0.2

    error = np.ones((Np, Np))
    error[(Np-1)//2, (Np-1)//2] = 10
    #plt.imshow(field)
    #plt.show()

    return ratio, error

def gaussian_field():
    """Generation of a Gaussian Kernel centered on point (X,Y)"""

    k1d = signal.gaussian(Np, std=10).reshape(Np, 1)
    kernel = np.outer(k1d, k1d)

    A = np.zeros((Np, Np))
    A[Np//2-(Np//2):Np//2+(Np//2)+1, Np//2-(Np//2):Np//2+(Np//2)+1] = kernel

    A = A*10
    A[(Np-1)//2, (Np-1)//2] = 2

    return A

field = diagonal_field()
#field = gaussian_field()
ratio, error = error_field()

# Compute spatial correlations
coords = [(lon/100., lat/100.) for lat in range(Np) for lon in range(Np)]
codist = Preprocessing_ANTILOPE.codistances(coords, ld=0.02)
#cd     = codist.toarray()

# Add error ponderation
pond = codist.dot(diags(np.exp(-(error-1)).flatten(), 0))  # error is in [1, inf[
#pond[24,24] pour le pixel central

# Compute new field
new = Preprocessing_ANTILOPE.dynamic_correction(field, pond)
new = new.reshape((Np, Np))

plt.imshow(new)
plt.show()

import pdb
pdb.set_trace()

