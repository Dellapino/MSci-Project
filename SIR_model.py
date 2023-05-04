'''
The following is taken from 'making neuroSIR work.ipynb
'''



##################### LIBRARIES #################################################################################################################################################



import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import random
import networkx as nx
import os
import glob
from PIL import Image



##################### INITIALISATION #################################################################################################################################################



def initialise_potential(G, params): #look into more initialisation schemes
    nodes = list(G.nodes)
    
    scheme = params['Scheme']
    
    if scheme == 'random':
        for node in nodes:
            if random.random() < params['Initial']:
                G.nodes[node]['potential'] = params['Threshold']
            else:
                G.nodes[node]['potential'] = 0
    
    elif scheme == 'local':
        
        origin = random.choice(nodes)
        G.nodes[origin]['potential'] = params['Threshold'] 
        
        # compute the shortest path lengths from the start node to all other nodes
        distances = nx.shortest_path_length(G, source=origin)

        # sort the nodes by their distance from the start node
        sorted_nodes = sorted(distances.items(), key=lambda x: x[1])

        # compute the number of nodes that represent 5% of the total number of nodes in the graph
        num_nodes = len(G.nodes)
        num_closest_nodes = int(num_nodes * params['Initial'])

        # extract the closest nodes
        closest_nodes = [node for node, distance in sorted_nodes[num_nodes - num_closest_nodes:]]
        
        for node in nodes:
            G.nodes[node]['potential'] = 0
        
        for node in closest_nodes:
            G.nodes[node]['potential'] = params['Threshold']
        
    else:
        print('POTENTIAL NOT INITIALISED')
            
    return G

def initialise_status(G): # Assigns node a initial status
    '''
    1   :   susceptible
    0   :   infected
    -1  :   removed
    '''
    nodes = list(G.nodes)
    for node in nodes:
        if G.nodes[node]['potential'] == 0:
            G.nodes[node]['status'] = 1
        else:
            G.nodes[node]['status'] = 0
    return G

def initialise_weight(G, weight): # Sets weight to all edges to 'weight'
    nodes = list(G.nodes)
    for node in nodes:
        connections = list(G.neighbors(node))
        for connection in connections:
            G[node][connection]['weight'] = weight
    return G



#####################  PROPAGATION #################################################################################################################################################




def propagate(G, params): 
    nodes = list(G.nodes)
    
    update_dict = {}
    for node in nodes: # initialise dict to keep track of which nodes need to be updated and by how much
        update_dict[node] = 0
    
    for node in nodes: # finding which potential to increase
        if G.nodes[node]['status'] == 0: # if node is infected it should transmit disease and have prob of becoming recovered, first as these are actors
            connections = list(G.neighbors(node))
            for connection in connections: # spread infection to all neighbours
                update_dict[connection] += params['Beta'] * G[node][connection]['weight'] 
            
            if np.random.random() < params['Gamma']: # some chance for infected nodes to become removed nodes
                G.nodes[node]['status'] = -1
    
    for node in nodes: # carrying out updates by looping over all susceptible nodes to increase potentials
        if G.nodes[node]['status'] == 1: # if a node is susceptible, it should be able to gain potential and can become infected
            G.nodes[node]['potential'] += update_dict[node]
            if G.nodes[node]['potential'] >= params['Threshold']: # if update pushes above threshold then infected
                G.nodes[node]['status'] = 0
            else: # otherwise there is decay
                pot = G.nodes[node]['potential']
                
                decay = pot * np.exp(-pot / params['Threshold'])
                
                G.nodes[node]['potential'] -= decay

    return G



#####################  MEASUREMENTS #################################################################################################################################################



def check_activity(G, threshold): # cumsum of infected nodes
    infected = 0
    nodes = list(G.nodes)
    for node in nodes:
        if G.nodes[node]['potential'] >= threshold:
            infected += 1
    return infected / len(G.nodes)

def check_activity(G): # cumsum of infected nodes
    total = 0
    nodes = list(G.nodes)
    for node in nodes:
        if  G.nodes[node]['status'] == 0:
            total+=1
    return total / len(G)

def check_removed(G):
    total = 0
    nodes = list(G.nodes)
    for node in nodes:
        if  G.nodes[node]['status'] == -1:
            total+=1
    return total / len(G)

def check_susecptible(G): # finds what fraction of nodes are susceptible
    total = 0
    nodes = list(G.nodes)
    for node in nodes:
        if  G.nodes[node]['status'] == 1:
            total+=1
    return total / len(G)

def check_states(G):
    s = check_susecptible(G)
    i = check_activity(G)
    r = check_removed(G)
    return s, i, r



#####################  RUNNING #################################################################################################################################################



def simulate(G, params): # add in infected per time step later
    susceptible = []
    infected = []
    removed = []
    
    G = initialise_potential(G, params)
    G = initialise_status(G)
    
    s, i, r = check_states(G)
    susceptible.append(s)
    infected.append(i)
    removed.append(r)
    
    for t in range(params['Time']-1):
        G = propagate(G, params)
        s, i, r = check_states(G)
        susceptible.append(s)
        infected.append(i)
        removed.append(r)
        
    return susceptible, infected, removed

def smooth(G, params):
    s_smooth = []
    i_smooth = []
    r_smooth = []
    s_runs = []
    i_runs = []
    r_runs = []
    T = params['Time']
    M = params['Runs']
    for i in range(M):
        #print(str(round((i+1)*100/M, 1)) + '%') # display progress of smoothing
        s_run, i_run, r_run = simulate(G, params)
        s_runs.append(s_run)
        i_runs.append(i_run)
        r_runs.append(r_run)
    
    for i in range(T): # smoothing susceptible time series
        total = 0
        for j in range(M):
            total += s_runs[j][i]     
        total = total / M   
        s_smooth.append(total)
        
    for i in range(T): # smoothing infected time series
        total = 0
        for j in range(M):
            total += i_runs[j][i]     
        total = total / M   
        i_smooth.append(total)

    for i in range(T): # smoothing recovered time series
        total = 0
        for j in range(M):
            total += r_runs[j][i]     
        total = total / M   
        r_smooth.append(total)
        
    return s_smooth, i_smooth, r_smooth



##################### ANALYTICAL #################################################################################################################################################



def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def analytic_sol(G, params):
    N = len(G)
    I0 = np.ceil(params['Initial'] * N)
    R0 = 0
    S0 = N - I0 - R0
    t = np.arange(0, params['Time'], 1)
    y0 = S0, I0, R0 # Initial conditions vector
    ret = odeint(deriv, y0, t, args=(N, params['Beta'], params['Gamma']))
    S, I, R = ret.T
    return S/N, I/N, R/N



def comparison(G, params, name = 'Placeholder'):
    plt.figure(figsize=(8, 6))
    plt.grid()
    #plt.ylim(0, 1.2)
    analytic_time = np.arange(0, params['Time'])
    S, I, R = analytic_sol(G, params)

    plt.plot(analytic_time, S, linewidth = 3, label = 'Susceptible', color = 'b', linestyle = 'dashed', alpha = 1)
    plt.plot(analytic_time, I, linewidth = 3, label = 'Infected', color = 'r', linestyle = 'dashed', alpha = 1)
    plt.plot(analytic_time, R, linewidth = 3, label = 'Recovered', color = 'g', linestyle = 'dashed', alpha = 1)
    
    #eqn_infected = np.cumsum(I)
    #plt.plot(analytic_time, eqn_infected, linewidth = 3, label = 'Infected', color = 'r')
    
    sim_time = np.arange(0, params['Time'])
    s_sim, i_sim, r_sim = smooth(G, params['Initial'], params['Threshold'], params['Time'], params['Runs'], params['Beta'], params['Gamma'])
    

    plt.plot(sim_time, s_sim, color = 'b')
    plt.plot(sim_time, i_sim,  color = 'r')
    plt.plot(sim_time, r_sim,  color = 'g')
    
    #sim_infected = np.cumsum(i_sim)
    #plt.plot(sim_time, sim_infected, label = name)
    
    plt.legend(loc = 'center right')
    plt.title('Neuron SIR (beta = ' + str(params['Beta']) + ') (Gamma = ' + str(params['Gamma']) + ') (Threshold = ' + str(params['Threshold']) + ')')
    plt.show()
       
    #print('Final simulated infected: ' + str(sim_infected[-1]))
    #print('Final analytic infected: ' + str(eqn_infected[-1]))
    
    return G


##################### VISUALISATION #################################################################################################################################################



def find_colours(G):
    colours = []
    for node in list(G.nodes):
        if G.nodes[node]['status'] == 1:
            colours.append('blue')
        elif G.nodes[node]['status'] == 0:
            colours.append('red')
        elif G.nodes[node]['status'] == -1:
            colours.append('green')
        else:
            colours.append('purple')
    return colours

def visualise(G, params, seed_val):
    
    files = glob.glob('/Users/ali/MSci Project/IF visualisation/frame*.png')
    for f in files:
        os.remove(f)
    
    G = initialise_potential(G, params['Initial'], params['Threshold'])
    G = initialise_status(G)
    plt.figure(figsize = (12, 8))
    if seed_val == None:
        positions = nx.spring_layout(G)
    else:
        positions = nx.spring_layout(G, seed = seed_val)
    
    colours = find_colours(G)
    plt.text(10,10,'time')
    nx.draw_networkx(G, pos = positions, with_labels = 0, node_size = 100, node_color = colours, alpha = 0.5)
    plt.savefig('/Users/ali/MSci Project/IF visualisation/frame' + str(0))
    plt.clf()
    
    for t in range(params['Time']):
        G = propagate(G, params['Threshold'], params['Beta'], params['Gamma'])
        colours = find_colours(G)
        nx.draw_networkx(G, pos = positions, with_labels = 0, node_size = 100, node_color = colours, alpha = 0.5)
        plt.savefig('/Users/ali/MSci Project/IF visualisation/frame' + str(t+1))
        plt.clf()
        
def make_gif():
    frames = []
    imgs = []
    path = '/Users/ali/MSci Project/IF visualisation/frame'
    total_frames = len(os.listdir('/Users/ali/MSci Project/IF visualisation/'))
    for i in range(total_frames):
        imgs.append(path + str(i) + '.png')
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save('/Users/ali/MSci Project/IF visualisation.gif', 
                format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=500) # duration of each frame in milliseconds!
    return frames