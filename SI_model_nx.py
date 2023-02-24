'''

Should contain all functions needed to simulate analytically and through model the SI model

Note: doesn't make sense to have 'SI' in funciton names when whole library is called 'SI_model'

'''

import network_manipulation as nm
import matplotlib.pyplot as plt
import numpy as np
import random

def initialise_potential(G, initial, threshold): #look into more initialisation schemes
    nodes = list(G.nodes)
    for node in nodes:
        if random.random() < initial:
            G.nodes[node]['potential'] = threshold + 1
        else:
            G.nodes[node]['potential'] = 0
    return G

def propagate(G, threshold, beta):
    nodes = list(G.nodes)
    for node in nodes:
        connections = G.neighbors(node)
        if G.nodes[node]['potential'] >= threshold:
            for connection in connections:
                if np.random.random() < beta:
                    G.nodes[connection]['potential'] += G[node][connection]['weight']
    return G

def activity(G, threshold):
    infected = 0
    nodes = list(G.nodes)
    for node in nodes:
        if G.nodes[node]['potential'] >= threshold:
            infected += 1
    return infected / len(G.nodes)

def simulate(G, initial, threshold, T, beta):
    activities = []
    G = initialise_potential(G, initial, threshold)
    for t in range(T):
        G = propagate(G, threshold, beta)
        activities.append(activity(G, threshold))
    return activities

def smooth(G, initial, threshold, T, M, beta = 0.6):
    smoothed = []
    runs = []
    for i in range(M):
        print(i)
        #G = initialise_potential(G, initial, threshold)
        run = simulate(G, initial, threshold, T, beta)
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

def generate_t(T, h): # generates time axis from 0 to T with increments of h, waste of space icl
    return np.arange(0, T, h)

def SI_comparison(G, params, name = 'Placeholder'):
    plt.figure()
    sim_activity = smooth(G, params['Initial'], params['Threshold'], params['Time'], params['Runs'], params['Beta'])
    sim_time = np.arange(0, params['Time'], 1)
    plt.plot(sim_time, sim_activity, label = name)
    
    analytic_time = np.arange(0, params['Time'], params['Increment'])
    const = integration_const(params['Initial'])
    analytic_activity = analytic_sol(analytic_time, params['Beta'], const)
    plt.plot(analytic_time, analytic_activity, linewidth = 3, label = 'Analytical')
    
    plt.legend(loc = 'lower right')
    plt.title('SI model comparison (beta = ' + str(params['Beta'])+ ') (Threshold = ' + str(params['Threshold']) + ')')
    plt.show()


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