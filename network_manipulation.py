'''

Collection of funcitons needed to manipulate or extract information from networks

'''
import networkx as nx
import numpy as np

#__________________________________________________________________________________

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

# ___________________________________________________________________________

def check_unique(network):
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
    if nodes == len(vals):
        return True
    else:
        return False

def find_size(network):
    return len(network)

def find_connections(network):
    nodes = list(network.keys())
    connections = []
    for i in range(len(network)):
        connections.append(network[nodes[i]][0])
    return connections

def find_strengths(network, normed = True):
    nodes = list(network.keys())
    strengths = []
    for i in range(len(network)):
        strengths.append(network[nodes[i]][1])
    if normed:
        norm = max(max(s) for s in strengths)
        for i in range(len(network)):
            for j in range(len(strengths[i])):
                strengths[i][j] = strengths[i][j] / norm
    return strengths

def find_time(path, sentence = False):
    data = access_dataset(path)
    start = int(data[0][0])
    end = int(data[-1][0])
    time = round((end - start) / 60 / 60 / 24, 1)
    if sentence:
        print('The dataset is built from interactions over ' + str(time) + ' days')
    return time


# ________________ making code compatible with networkx____________________

'''
NetworkX can generate graphs from dict of dict input which is what we'll use:
d = {0: {1: {"weight": 1}}}  # dict-of-dicts single edge (0,1)
G = nx.Graph(d)
'''

# tried to build a complex data structure of a dict of dict of dict but didn't work
def build_graph(name):
    network = build_network(name)
    nodes = list(network.keys())
    graph = {}
    #for node in nodes:
    node = nodes[0]
    temp = {network[node][0][0] : {'weight' : network[node][1][0]}}    
    for j in range(1, len(network[node][0])):
        #temp = {}
        #temp[network[node][0][j]] = {'weight' : network[node][1][j]}
        temper = {'weight' : network[node][1][j]}
        temp.update(temper)
        print(temp)
    print('\n')
    graph[node] = temp # might need to make keys into integers
    return graph

# try use G.add_edge(2, 3, {'weight': 3.1415}) to build graph, WORKS!
def build_nxgraph(name, normed = True):
    network = build_network(name)
    nodes = list(network.keys())
    graph = nx.Graph()
    connections = find_connections(network)
    strengths = find_strengths(network, normed)
    for i in range(len(nodes)):
        for j in range(len(connections[i])):
            graph.add_edge(nodes[i], connections[i][j], weight = strengths[i][j])
    return graph


# _______________building larger network with same properties____________________

'''
Need to convert this to work with new network definition and be able to generate
realistic weights in an arbitrarily sized network
'''

def scale_degree_dist(degrees):
    return False

def avg_degree_intel(graph_dict):
    size = len(graph_dict)
    nodes = list(graph_dict.keys())
    degree_dist = []
    for i in range(size):
        degree_dist.append(len(graph_dict[nodes[i]]))
    avg_degree = sum(degree_dist) / size
    std_degree = np.std(degree_dist)
    return avg_degree, std_degree

def degree_dist(network):
    '''
    Extract degree dist from a network
    '''
    size = find_size(network)
    nodes = list(network.keys())
    degrees = []
    for i in range(size):
        degrees.append(len(network[nodes[i]][0]))
    return degrees

def degree_dist_stats(dist):
    '''
    Takes a degree dist as input and finds avg degree and standard deviation
    '''
    avg = sum(dist) / len(dist)
    std = np.std(dist)
    return avg, std    
        
def generate_degree_dist(avg, std, length):
    '''
    Generates a degree distribution according to normal distribution
    Should perhaps use log-normal?
    Or perhaps maxwell-boltzmann?
    '''
    degree_dist = []
    while len(degree_dist) < length - 1:
        degree = round(np.random.normal(avg, std))
        if degree >= 0:
            degree_dist.append(degree)
    if sum(degree_dist) % 2 == 1:
        while True:
            degree = round(np.random.normal(avg, std))
            if degree % 2 == 1:
                degree_dist.append(degree)
                break
    else:
        while True:
            degree = round(np.random.normal(avg, std))
            if degree % 2 == 0:
                degree_dist.append(degree)
                break
    return degree_dist

def generate_degree_dist_v2(avg, std, length):
    '''
    Generates a degree distribution according to normal distribution
    need to round
    '''
    degree_dist = []
    while len(degree_dist) < length - 1:
        degree = np.random.normal(avg, std)
        degree_dist.append(degree)
    degree_dist = np.array(degree_dist)
    return degree_dist

def generate_degree_dist_v3(avg, std, length):
    '''
    Generates a degree distribution according to 
    '''
    degree_dist = []
    while len(degree_dist) < length - 1:
        degree = np.random.normal(avg, std)
        if degree >= 0:
            degree_dist.append(degree)
    return degree_dist

def generate_degree_dist_v4(avg, std, length):
    '''
    Generates a degree distribution according to log-normal
    '''
    degree_dist = []
    while len(degree_dist) < length - 1:
        degree = np.random.normal(avg, std)
        if degree >= 0:
            degree_dist.append(degree)
    return degree_dist

def generate_large_graph(graph_dict, scale, seed = None): # generates a graph with self loops but should not be a problem as an infected node cannot become any more infected that it already is however this may cause poor representation of graph connections, but this is only a problem for small graphs, so should hold for at least SI model in large graph limit
    a, s = avg_degree_intel(graph_dict)
    d = generate_degree_dist(a, s, scale)
    g = nx.configuration_model(d, nx.Graph, seed)
    return g

def adjacency_dict(nx_graph): 
    keys = list(nx_graph.nodes())
    adjacency_dict = {}
    for i in range(len(keys)):
        adjacency_dict[keys[i]] = [n for n in nx_graph.neighbors(keys[i])]
    return adjacency_dict


# find average weight / distribution of weights + other network measures