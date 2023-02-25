import matplotlib.pyplot as plt
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

def initialise_status(G):
    nodes = list(G.nodes)
    for node in nodes:
        G.nodes[node]['active'] = True
    return G

def propagate(G, threshold, beta): # only fires if newly infected, no decay of potential
    nodes = list(G.nodes)
    for node in nodes:
        if G.nodes[node]['active']: # if True i.e. node hasn't fired previously
            if G.nodes[node]['potential'] >= threshold:
                G.nodes[node]['active'] = False
                connections = G.neighbors(node)
                for connection in connections:
                    if np.random.random() < beta:
                        G.nodes[connection]['potential'] += G[node][connection]['weight']
    return G
    
def activity(G, threshold): # cumsum of infected nodes
    infected = 0
    nodes = list(G.nodes)
    for node in nodes:
        if G.nodes[node]['potential'] >= threshold:
            infected += 1
    return infected / len(G.nodes)

def check_susecptible(G): # finds what fraction of nodes are susceptible
    total = 0
    nodes = list(G.nodes)
    for node in nodes:
        if  G.nodes[node]['active'] == True:
            total+=1
    return total / len(G)

def firing(G): # can work this out by difference in cumsum!
    return G

def simulate(G, initial, threshold, T, beta): # add in infected per time step later
    activities = []
    G = initialise_potential(G, initial, threshold)
    G = initialise_status(G)
    activities.append(activity(G, threshold))
    for t in range(T):
        G = propagate(G, threshold, beta)
        activities.append(activity(G, threshold))
        #print(check_susecptible(G))
    return activities

def smooth(G, initial, threshold, T, M, beta = 0.6):
    smoothed = []
    runs = []
    for i in range(M):
        #print(str(round((i+1)*100/M, 1)) + '%') # display progress of smoothing
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

def comparison(G, params, name = 'Placeholder'):
    plt.figure()
    analytic_time = np.arange(0, params['Time'], params['Increment'])
    const = integration_const(params['Initial'])
    analytic_activity = analytic_sol(analytic_time, params['Beta'], const)
    plt.plot(analytic_time, analytic_activity, linewidth = 3, label = 'Analytical')
    
    sim_activity = smooth(G, params['Initial'], params['Threshold'], params['Time'], params['Runs'], params['Beta'])
    sim_time = np.arange(0, params['Time'], 1)
    plt.plot(sim_time, sim_activity, label = name)
    
    plt.legend(loc = 'lower right')
    plt.title('SI model comparison (beta = ' + str(params['Beta'])+ ') (Threshold = ' + str(params['Threshold']) + ')')
    plt.show()
    
    return G