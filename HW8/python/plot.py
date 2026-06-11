import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

L = np.array([8, 10, 12, 14, 16, 18, 20, 22, 24])
beta = np.array([0.51818, 0.52932, 0.53562, 0.53967, 0.54232, 0.54421, 0.54533, 0.54649, 0.54726])
err = np.array([0.00010, 0.00010, 0.00011, 0.00010, 0.00008, 0.00009, 0.00008, 0.00005, 0.00007])

L_tilde = 1 / L**2

y = np.sum(beta/(err**2))
x = np.sum(L_tilde/(err**2))
sigma = np.sum(1/(err**2))
x2 = np.sum(L_tilde**2/(err**2))
xy = np.sum(L_tilde*beta/(err**2))

b = (y*x - sigma*xy) / (x**2-sigma*x2)
a = (y - b*x) / sigma

L_pred = np.linspace(8, 24, 100)
beta_pred = a + b / L**2

chi2 = np.sum(((beta - (a + b / L**2)) / err)**2)
chi2_red = chi2 / (len(L) - 2)
error = (beta - beta_pred) / err
print(f'beta: {beta}, beta_pred: {beta_pred}, error: {error}')

