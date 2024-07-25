# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 23:13:08 2024

@author: eoyur
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Parameters
N = 100  # Number of players
m = 3  # Number of connections each new node makes in the Barabási–Albert network
alpha = 0.1  # Altruism parameter
beta = 0.5  # Benefit factor from the public good
gamma = 0.01  # Learning rate
T = 100  # Number of time steps

# Initialize Barabási–Albert network
G = nx.barabasi_albert_graph(N, m)
adj_matrix = nx.to_numpy_array(G)

# Initial contributions
c = np.random.rand(N)

# Simulation
for t in range(T):
    G_total = np.sum(c)
    for i in range(N):
        N_i = np.where(adj_matrix[i
