'''
Contains functions to convert interaction data from sociopatterns datasets to a dictionary that stores a weighted undirected network:

{
'node': [['edge', 'edge', 'edge'], ['weight', 'weight', 'weight']]
}


Consider: seperate dictionaries for connections and weights instead of extracting this from one dictionary,
will allow code to be resused easily for unweighted networks, alternatively can just set all weights to 1,
this sounds like a better solution as all network information is in one object.
'''

import numpy as np

def access_dataset(name):
    data = open(name, 'r')
    lines = data.readlines() 
    lines = [line.split() for line in lines] # 2D array containing lines from data, split into words stored as strings
    return lines

def build_dict_1(network, lines):
    for i in range(len(lines)): # This works! but generates a directed network
        a = lines[i][1] # first person in interaction
        b = lines[i][2] # second person in interaction
        
        try: # editing an existing node
            if b in network[a][0]:
                index = network[a][0].index(b)
                network[a][1][index] += 1 # increasing interaction strength
            else:
                network[a][0].append(b)
                network[a][1].append(1)
        except: # adding new interaction
            network[a] = [[b], [1]]
    return network

def build_dict_2(network, lines):
    for i in range(len(lines)): # This works! but generates a directed network
        b = lines[i][1] # first person in interaction
        a = lines[i][2] # second person in interaction
        
        try: # editing an existing node
            if b in network[a][0]:
                index = network[a][0].index(b)
                network[a][1][index] += 1 # increasing interaction strength
            else:
                network[a][0].append(b)
                network[a][1].append(1)
        except: # adding new interaction
            network[a] = [[b], [1]]
    return network 
    
def build_network(name):
    lines = access_dataset(name)
    network = {}
    network = build_dict_1(network, lines)
    network = build_dict_2(network, lines)
    return network

def find_size(network):
    vals = network.keys()
    test_unique = []
    for i in vals:
        if i in test_unique:
            pass
        else:
            test_unique.append(i)
    nodes = len(test_unique)
    print('Number of nodes: ' + str(len(vals)))
    print('Number of unique nodes: ' + str(nodes))
    return nodes

def find_connnections(network):
    nodes = list(network.keys())
    connections = []
    for i in range(len(network)):
        connections.append(network[nodes[i]][0])
    return connections

def find_strengths(network):
    nodes = list(network.keys())
    strengths = []
    for i in range(len(network)):
        strengths.append(network[nodes[i]][1])
    norm = max(max(s) for s in strengths)
    for i in range(len(network)):
        for j in range(len(strengths[i])):
            strengths[i][j] = strengths[i][j] / norm
    return strengths

def find_time(path):
    data = access_dataset(path)
    start = data[0][0]
    end = data[-1][0]
    time = round((1022380 - 28840) / 60 / 60 / 24, 1)
    print('The dataset is built from interactions over ' + str(time) + ' days')
    return time