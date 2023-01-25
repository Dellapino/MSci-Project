'''

Should contain all functions needed to simulate analytically and through model the SI model

'''

import network_manipulation as nm
import numpy as np

def intitialise_potential(N, initial, threshold): # initial defines the fraction of populations that is initially infected
    P = {} # another graph to keep track of potential
    nodes = N.keys()
    for node in nodes:
        if random.random() < initial:
            P[node] = threshold
        else:
            P[node] = 0
    return P

def propagate_SI(P, N, threshold, strength = 1, beta = 0.6): # add checks for valid parameter values
    connections = nm.find_connections(N)
    strengths = nm.find_strengths(N)
    potentials = list(P.values())
    size = len(N)
    F = np.zeros(size)
    for i in range(size): # loop over every node and check if firing
        if potentials[i] >= threshold:
            if np.random.random() < beta:
                F[i] = 1
        for j in range(len(connections[i])):    
            if np.random.random() < beta: #0.6 is placeholder value for beta
                P[connections[i][j]] += strengths[i][j] * strength   
    return P, F


def propagate_SI_v2(P, N, threshold, strength = 1, beta = 0.6): # add checks for valid parameter values
    '''
    propagates contagion according to transmission prob and infection strength
    '''
    connections = nm.find_connections(N)
    strengths = nm.find_strengths(N)
    potentials = list(P.values())
    size = len(N)
    for i in range(size): # loop over every node and check if firing and propagate if needed
        if potentials[i] >= threshold:
            for j in range(len(connections[i])):    
                if np.random.random() < beta: 
                    P[connections[i][j]] += strengths[i][j] * strength   
    return P

def activity(P, threshold):
    '''
    Finds what fraction of population is infected
    '''
    potentials = list(P.values())
    size = len(potentials)
    activity = 0
    for i in range(size):
        if potentials[i] >= threshold:
            activity += 1
    return activity / size
    

def simulate_SI(N, P, T, h, threshold):
    size = len(N)
    activity = []
    limit = T 
    for t in range(limit):
        #time.sleep(1)
        P, F = weighted_propagate_SI(P, N, threshold)
        activity.append(sum(F) / size)
    return activity

def simulate_SI_v2(N, T, initial, threshold):
    '''
    Returns time series of infection activity up till time T
    '''
    activities = []
    P = intitialise_potential(N, initial, threshold)
    for t in range(T):
        P = propagate_SI_v2(P, N, threshold)
        activities.append(activity(P))
    return activities

def smooth_SI_v2(N, T, initial, threshold, M): # M is number of runs over which to smooth over
    '''
    Smoothes infection activity time series by averaging over M runs
    '''
    smoothed = []
    runs = []
    for i in range(M):
        P = intitialise_potential(N, initial, threshold)
        run = simulate_SI_v2(N, T, initial, threshold)
        runs.append(run)
    for i in range(T):
        total = 0
        for j in range(M):
            total += runs[j][i]     
        total = total / M   
        smoothed.append(total)
    return smoothed

def smooth_SI(N, I, T, M, h, threshold = 10): # M is number of runs over which to smooth over
    runs = []
    limit = int(T//h) + 1
    limit = T
    for i in range(M):
        P = intitialise_potential(N, I, threshold)
        run = simulate_SI(N, P, T, h, threshold)
        runs.append(run)
    smoothed = []
    for i in range(limit):
        total = 0
        for j in range(M):
            total += runs[j][i]     
        total = total / M   
        smoothed.append(total)
    return smoothed

#______________________`siu ______________________`


def SI_diff(initial, beta, T, h): # differential equation representing si model with N total nodes
    I = initial # intial fraction of infected population
    spread = []
    limit = int(T//h) + 1
    for t in range(limit): # can also run until population is fully infected
        I_new = h * beta * I * (1 - I) + I# taking N = 1 as total population
        I = I_new
        spread.append(I_new)
    return spread


#______________________`siu ______________________`

def analytic_sol(x, beta, c):
    return np.exp(beta*x+c) / (1 + np.exp(beta*x+c))

def integration_const(I):
    return np.log(I/(1-I))

def generate_t(T, h): # generates time axis from 0 to T with increments of h
    return np.arange(0, T, h)



# EXPERIMENTAL

from scipy.optimize import curve_fit

def SI(time, beta, h):
    I_old = 0.05
    for i in range(time):
        I_new = h * beta * I_old * (1 - I_old) + I_old
        I_old = I_new
    return I_old

smoothed_SI = np.array(smoothed_SI)

popt, pcov = curve_fit(SI, t, smoothed_SI)



# TO DO

# function which takes input of all necessary parameters and returns either or all of model or analytical solution
# should use the visualisation library to display these iA