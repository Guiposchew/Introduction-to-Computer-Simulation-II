import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

T_c = 2.269185314213

A = pd.read_csv('HW2/data_prof/E_8.csv')
B = pd.read_csv('HW2/data_prof/E_16.csv')
C = pd.read_csv('HW2/data_prof/E_32.csv')
D = pd.read_csv('HW2/data_prof/E_64.csv') 

X = pd.read_csv('HW2/data_prof/specific_heat_L8.csv')
Y = pd.read_csv('HW2/data_prof/specific_heat_L16.csv')
Z = pd.read_csv('HW2/data_prof/specific_heat_L32.csv')
W = pd.read_csv('HW2/data_prof/specific_heat_L64.csv')


beta_8=A['beta'].to_numpy()
E_8 = A['E'].to_numpy()

beta_16=B['beta'].to_numpy()
E_16 = B['E'].to_numpy()

beta_32=C['beta'].to_numpy()
E_32 = C['E'].to_numpy()

beta_64=D['beta'].to_numpy()
E_64 = D['E'].to_numpy()

beta_h_8=X['beta'].to_numpy()
Cv_8 = X['Cv'].to_numpy()

beta_h_16=Y['beta'].to_numpy()
Cv_16 = Y['Cv'].to_numpy()

beta_h_32=Z['beta'].to_numpy()
Cv_32 = Z['Cv'].to_numpy()

beta_h_64=W['beta'].to_numpy()
Cv_64 = W['Cv'].to_numpy()


plt.figure(1)
plt.plot(1/(beta_8*T_c), -(E_8), label='L = 8')
plt.plot(1/(beta_16*T_c), -(E_16), label='L = 16')
plt.plot(1/(beta_32*T_c), -(E_32), label='L = 32')
plt.plot(1/(beta_64*T_c), -(E_64), label='L = 64')
plt.xlim(0.75, 1.3)
plt.ylim(-1.90,-0.85)
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$\langle E \rangle / N$')
plt.title('Energy per spin for different lattice sizes')
plt.legend()

plt.figure(2)
plt.plot(1/(beta_h_8*T_c), Cv_8, label='L = 8')
plt.plot(1/(beta_h_16*T_c), Cv_16, label='L = 16')
plt.plot(1/(beta_h_32*T_c), Cv_32, label='L = 32')
plt.plot(1/(beta_h_64*T_c), Cv_64, label='L = 64')
plt.xlim(0.1,1.9)
plt.ylim(-0.1, 3.6)
plt.xlabel(r'$\beta/\beta_c$')
plt.ylabel(r'$C/ N$')
plt.title('Specific Heat for different lattice sizes')
plt.legend()
plt.show()