# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 23:04:46 2024

@author: eoyur
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# Parameters
N = 100  # Number of players
p = 0.1  # Connection probability
alpha = 0.1  # Altruism parameter
beta = 0.5  # Benefit factor from the public good
gamma = 0.01  # Learning rate
T = 100  # Number of time steps

# Initialize Erdős–Rényi network
G = nx.erdos_renyi_graph(N, p)
adj_matrix = nx.to_numpy_array(G)

# Initial contributions
c = np.random.rand(N)

# Simulation
for t in range(T):
    G_total = np.sum(c)
    for i in range(N):
        N_i = np.where(adj_matrix[i] == 1)[0]
        U_i = (1 - alpha) * (1 - c[i] + beta * G_total) + alpha * np.sum(1 - c[N_i] + beta * G_total)
        marginal_utility = -(1 - alpha) + beta * (1 - alpha + alpha * len(N_i))
        c[i] = c[i] + gamma * marginal_utility

# Analysis
plt.figure(figsize=(10, 6))
plt.hist(c, bins=20, alpha=0.75)
plt.title('Distribution of Contributions after Simulation')
plt.xlabel('Contribution')
plt.ylabel('Frequency')
plt.show()
