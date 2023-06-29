# %%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy
from collections import defaultdict


def modulator(x, y):
    out = pow(np.cos(x + y - np.pi/4), 2) - 0.5
    return out


def feedback(x, alpha, beta, J):
    fb = beta * J
    np.fill_diagonal(fb, alpha)
    fb_signal = np.dot(fb, x)
    return fb_signal


def Ising_Energy(x, J):
    A = np.dot(J, x)
    B = np.dot(x.T, A)
    energy = -0.5 * B
    return energy


def normal(amplitude):
    for i,val in enumerate(amplitude):
        if(val > 0):
            amplitude[i] = 1
        elif(val < 0):
            amplitude[i] = -1
    return amplitude

N_node = 10
N_spin = pow(N_node, 2) 
j = defaultdict(int)
J = np.zeros([N_spin, N_spin])
G = nx.Graph()
with open(r"/workspace/Attempt_optimizer/d-w/output_500.txt") as f:
    lines = f.readlines()
    for line in lines:
        l = line.split()
        j[(int(l[0]),int(l[1]))] = int(l[2])
        r,c,w = int(l[0])-1, int(l[1])-1, float(l[2])
        J[r][c] = w
        J[c][r] = w
G.add_edges_from(j)

K_inter = 1000
alpha = 0.25
beta = 0.29
x_k = np.zeros([N_spin, 1])
x_f = np.zeros([N_spin, 1])
energy = np.zeros([K_inter + 1, 1])
bifur = np.zeros([N_spin, 1])
noise = np.random.normal(0, 0.01, [N_spin, K_inter])

for i in range(K_inter):
    x_f = feedback(x_k, alpha, beta, J)
    x_k = modulator( x_f, np.array([noise[:, i]]).T )
    x_tem = x_k.copy()
    x_tem = normal(x_tem)
    energy[i+1] = Ising_Energy(x_tem, J)
    bifur = np.c_[bifur, x_k]

up = np.sum(x_k > 0)
down = np.sum(x_k < 0)
print("Up {}, down: {}".format(up, down))
plt.figure(figsize=(10, 8), dpi=120)
plt.subplot(2, 1, 1)
plt.ylabel("Amplitude//N")
for i in range(N_spin):
    plt.plot(bifur[i])
plt.subplot(2, 1, 2)
plt.title("Energy")
plt.plot(energy, '-')
plt.ylabel("Energy")
plt.show()
# %%
