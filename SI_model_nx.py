'''

Should contain all functions needed to simulate analytically and through model the SI model

Note: doesn't make sense to have 'SI' in funciton names when whole library is called 'SI_model'

'''

import network_manipulation as nm
import numpy as np
import random

def initialise_potential(G, initial, threshold): #look into more initialisation schemes
    nodes = list(G.nodes)
    for node in nodes:
        if random.random() < initial:
            G.nodes[node]['potential'] = threshold
        else:
            G.nodes[node]['potential'] = 0
    return G

def propagate(G, threshold, transmission = 1, beta = 0.6):
    nodes = list(G.nodes)
    for node in nodes:
        connections = G.neighbors(node)
        if G.nodes[node]['potential'] >= threshold:
            for connection in connections:
                if np.random.random() < beta:
                    G.nodes[connection]['potential'] += transmission * G[node][connection][0]['weight']
    return G

def activity(G, threshold):
    infected = 0
    nodes = list(G.nodes)
    for node in nodes:
        if G.nodes[node]['potential'] > threshold:
            infected += 1
    return infected / len(G.nodes)

def simulate(G, initial, threshold, T):
    activities = []
    G = initialise_potential(G, initial, threshold)
    for t in range(T):
        G = propagate(G, threshold)
        activities.append(activity(G, threshold))
    return activities

def smooth(G, initial, threshold, T, M):
    smoothed = []
    runs = []
    for i in range(M):
        print(i)
        G = initialise_potential(G, initial, threshold)
        run = simulate(G, initial, threshold, T)
        runs.append(run)
    for i in range(T):
        total = 0
        for j in range(M):
            total += runs[j][i]     
        total = total / M   
        smoothed.append(total)
    return smoothed

def analytic_sol(x, beta, c):
    return np.exp(beta*x+c) / (1 + np.exp(beta*x+c))

def integration_const(initial):
    return np.log(initial/(1-initial))

def generate_t(T, h): # generates time axis from 0 to T with increments of h
    return np.arange(0, T, h)



# _________________EXPERIMENTAL_________________

'''
Optimize parameters to match model and analytic solution
'''

from scipy.optimize import curve_fit



# TO DO

# function which takes input of all necessary parameters and returns either or all of model or analytical solution
# should use the visualisation library to display these iA

# Look at initialising in a non-discrete fashion -> have some at bit below threshold etc,
# then can compare how initial condition affect final result more cleanly