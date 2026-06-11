import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from main import specific_heat, potts_energy
from matplotlib.ticker import NullLocator, ScalarFormatter
 
beta = []
L = []
mean_cluster_size = []
fits = []
q=2

for i in range(1,11):
    df= pd.read_csv(f'HW5/python/data/data_q{(i/10):.2f}.csv')
    L.append(df['L'].to_numpy())
    beta.append(df['Beta'].to_numpy()[0])
    mean_cluster_size.append(df['mean_cluster_size'].to_numpy())

beta = np.array(beta)
L = np.array(L[0])
mean_cluster_size = np.array(mean_cluster_size)
susceptibility_w = []

for i in range(len(L)):
    susceptibility_w.append(mean_cluster_size[:, i]*beta** (1 + (q / (q - 1)) /(L[i]**2) ))

susceptibility_w = np.array(susceptibility_w)

susceptibility_w = susceptibility_w.reshape(10, 3)

susceptibility_m =[]

for i in range(1,11):
    df= pd.read_csv(f'HW5/python/data/data_metropolis_q{(np.log(1 + np.sqrt(2))*i/10):.2f}.csv')
    susceptibility_m.append(df['susceptibility'].to_numpy())




for i in range(10):
    plt.figure(i)
    plt.title(f'Susceptibility vs L for beta = {(beta[i]/beta[-1]):.2f}')
    plt.plot(L, susceptibility_w[i])
    plt.plot(L, susceptibility_m[i])
    plt.xlabel('L')
    plt.ylabel('Susceptibility')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
plt.show()