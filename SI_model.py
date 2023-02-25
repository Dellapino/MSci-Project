'''

Should contain all functions needed to simulate analytically and through model the SI model

Note: doesn't make sense to have 'SI' in funciton names when whole library is called 'SI_model'

'''

import network_manipulation as nm
import numpy as np
import random

def initialise_potential(N, initial, threshold): # initial defines the fraction of populations that is initially infected
    '''
    Randomly infects a given fraction of the popualtion
    '''
    P = {} # another graph to keep track of potential
    nodes = N.keys()
    for node in nodes:
        if random.random() < initial:
            P[node] = threshold
        else:
            P[node] = 0
    return P
        
def propagate(P, N, threshold, strength = 1, beta = 0.6): # add checks for valid parameter values
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

def simulate(N, T, initial, threshold):
    '''
    Returns time series of infection activity up till time T
    '''
    activities = []
    P = initialise_potential(N, initial, threshold)
    for t in range(T):
        P = propagate(P, N, threshold)
        activities.append(activity(P, threshold))
    return activities

def smooth(N, T, initial, threshold, M): # M is number of runs over which to smooth over
    '''
    Smoothes infection activity time series by averaging over M runs
    '''
    smoothed = []
    runs = []
    for i in range(M):
        P = intitialise_potential(N, initial, threshold)
        run = simulate(N, T, initial, threshold)
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