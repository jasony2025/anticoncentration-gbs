import numpy as np
from scipy.stats import unitary_group
import math
import sympy
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

print("hi")

N = 200
s = 0.1
num = 100

x = [i/N for i in range(1, N)]
y_all = []


for i in range(num):
    print("run no " + str(i+1))
    
    U = unitary_group.rvs(N) # U is NxN

    y_curr = []
    
    for r in x:
        m = round(r*N)
        n = N-m
        
        T = np.matrix(np.diag([1 if i < 2 * m or i % 2 == 0 else -1 for i in range(2 * N)])) # checked, correct
        
        
        Ubar = np.conj(U)
        Udag = np.conj(U).T
        
        # M = np.matrix(np.vstack((np.hstack((np.real(Ubar@Udag), np.imag(Ubar@Udag))), np.hstack((np.imag(Ubar@Udag), -1*np.real(Ubar@Udag))))))
        M = np.matrix(np.vstack((np.hstack((np.real(U@U.T), np.imag(U@U.T))), np.hstack((np.imag(U@U.T), -1*np.real(U@U.T))))))
        Mtilde = T@M@T
        
        omega = np.kron(np.eye(N), np.array([[0, 1], [-1, 0]])) # checked, correct
    
        sigma = np.cosh(2*s) * np.identity(2*N) + np.sinh(2*s) * M # checked, correct
        sigmaTilde = T@sigma@T
    
        neg = 0
        
        for i in np.linalg.eig(1j * omega@sigmaTilde)[0]:
            neg += max(0, -math.log(abs(i)))
        
        y_curr.append(neg/2)

    y_all.append(y_curr)

y_av = np.mean(np.array(y_all), axis=0)
y_smooth = savgol_filter(y_av, N-1, 2)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

for y_run in y_all:
    ax.plot(x, y_run, color='cornflowerblue', alpha=0.4, linewidth=1.5)

ax.plot(x, y_smooth, color='crimson', linewidth=3, label='Smoothed Average of 100 Runs')
ax.plot([], [], color='cornflowerblue', alpha=0.6, label='Individual Unitary Runs') # Dummy plot for legend

ax.set_xlabel('Subsystem Fraction (r)', fontsize=14)
ax.set_ylabel('Negativity', fontsize=14)
ax.set_title(f'Negativity vs. Subsystem Fraction (N={N}, s={s})\n(10 Random Unitary Matrices)', fontsize=16)
ax.legend(fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.show()
