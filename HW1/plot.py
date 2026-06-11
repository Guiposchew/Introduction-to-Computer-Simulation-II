import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

A = pd.read_csv('data/ising_L8.csv')
B = pd.read_csv('data/ising_L16.csv')
C = pd.read_csv('data/ising_L32.csv')
D = pd.read_csv('data/ising_L64.csv')

beta_normal = A['beta_over_beta_c'].to_numpy()
U_A = A['binder'].to_numpy()

U_B = B['binder'].to_numpy()
U_C = C['binder'].to_numpy()
U_D = D['binder'].to_numpy()

# plt.plot(1/beta_normal, U_A, marker='o', linestyle='-', label='L=8')
# plt.plot(1/beta_normal, U_B, marker='o', linestyle='-', label='L=16')
# plt.plot(1/beta_normal, U_C, marker='o', linestyle='-', label='L=32')
# plt.plot(1/beta_normal, U_D, marker='o', linestyle='-', label='L=64')
# plt.xlabel(r'$T / T_c$')
# plt.ylabel('Binder Parameter')
# plt.title('Binder Parameter vs Normalized Temperature')
# plt.legend()
# plt.show()

dif1 = (U_A-U_B)**2
dif2 = (U_A-U_C)**2
dif3 = (U_A-U_D)**2
dif4 = (U_B-U_C)**2
dif5 = (U_B-U_D)**2
dif6 = (U_C-U_D)**2

dif = np.abs(dif1 + dif2 + dif3 + dif4 + dif5 + dif6)/6

# # plt.plot(1/beta_normal, dif1, marker='o', linestyle='-', label='L=8 vs L=16')
# # plt.plot(1/beta_normal, dif2, marker='o', linestyle='-', label='L=8 vs L=32')
# # plt.plot(1/beta_normal, dif3, marker='o', linestyle='-', label='L=8 vs L=64')
# # plt.plot(1/beta_normal, dif4, marker='o', linestyle='-', label='L=16 vs L=32')
# # plt.plot(1/beta_normal, dif5, marker='o', linestyle='-', label='L=16 vs L=64')
# #plt.plot(1/beta_normal, dif6, marker='o', linestyle='-', label='L=32 vs L=64')
# plt.plot(1/beta_normal, dif, marker='o', linestyle='-', label='Average Difference')
# plt.xlabel(r'$T / T_c$')
# plt.ylabel('Binder Parameter Difference')
# plt.title('Difference in Binder Parameter between Lattice Sizes')
# plt.legend()
# plt.show()

# idx_T_c = np.argmin(np.abs(1-beta_normal))
# U_A_T_c = U_A[idx_T_c]
# U_B_T_c = U_B[idx_T_c]      
# U_C_T_c = U_C[idx_T_c]
# U_D_T_c = U_D[idx_T_c]
# print(f"Estimated U at T_c: L=8: {U_A_T_c:.3f}, L=16: {U_B_T_c:.3f}, L=32: {U_C_T_c:.3f}, L=64: {U_D_T_c:.3f}")
# print(f"Mean U at T_c: {((U_A_T_c + U_B_T_c + U_C_T_c + U_D_T_c)/4):.3f}")

M_A = A['magnetization_abs'].to_numpy()
M_B = B['magnetization_abs'].to_numpy()
M_C = C['magnetization_abs'].to_numpy()
M_D = D['magnetization_abs'].to_numpy()

delta_A = np.mean((M_A**2))-np.mean(M_A)**2
delta_B = np.mean((M_B**2))-np.mean(M_B)**2 
delta_C = np.mean((M_C**2))-np.mean(M_C)**2
delta_D = np.mean((M_D**2))-np.mean(M_D)**2

print(delta_A, delta_B, delta_C, delta_D)