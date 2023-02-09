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