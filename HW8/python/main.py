import numpy as np
import matplotlib.pyplot as plt

L = np.array([8, 10, 12, 14, 16, 18, 20, 22, 24])
beta = np.array([0.51818, 0.52932, 0.53562, 0.53967,
                 0.54232, 0.54421, 0.54533, 0.54649, 0.54726])
err = np.array([0.00010, 0.00010, 0.00011, 0.00010,
                0.00008, 0.00009, 0.00008, 0.00005, 0.00007])

print(f"{'Lmin':>5} {'N':>3} {'a':>12} {'b':>12} {'chi2/dof':>12}")

for Lmin in [8, 10, 12, 14, 16, 18]:

    mask = L >= Lmin

    L_fit = L[mask]
    beta_fit = beta[mask]
    err_fit = err[mask]

    x = 1 / L_fit**2
    w = 1 / err_fit**2

    S = np.sum(w)
    Sx = np.sum(w * x)
    Sy = np.sum(w * beta_fit)
    Sxx = np.sum(w * x**2)
    Sxy = np.sum(w * x * beta_fit)

    Delta = S * Sxx - Sx**2

    a = (Sxx * Sy - Sx * Sxy) / Delta
    b = (S * Sxy - Sx * Sy) / Delta

    beta_pred_chi = a + b * x

    chi2 = np.sum(((beta_fit - beta_pred_chi) / err_fit)**2)

    dof = len(L_fit) - 2

    chi2_red = chi2 / dof if dof > 0 else np.nan

    print(f"{Lmin:5d} {len(L_fit):3d} {a:12.6f} {b:12.6f} {chi2_red:12.3f}")

    x_graph = np.linspace(0, 1 / L_fit.min()**2, 100)
    beta_pred = a + b * x_graph

    plt.plot(x**(-0.5), beta_fit, 'o', label='Data, Lmin = {}'.format(Lmin))
    plt.plot(x_graph**(-0.5), beta_pred, '-', label='Fit, Lmin = {}'.format(Lmin))

plt.xlabel(r'$1/L^2$')
plt.ylabel(r'$\beta$')
plt.title('Linear Fit of Beta vs. 1/L^2')
plt.legend()
plt.show()