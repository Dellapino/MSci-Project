'''

Collection of funcitons needed to manipulate or extract information from networks

'''

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

def find_connections(network):
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

# building larger network with same properties

def avg_degree_intel(graph_dict):
    size = len(graph_dict)
    nodes = list(graph_dict.keys())
    degree_dist = []
    for i in range(size):
        degree_dist.append(len(graph_dict[nodes[i]]))
    avg_degree = sum(degree_dist) / size
    std_degree = np.std(degree_dist)
    return avg_degree, std_degree
        
def generate_degree_dist(avg, std, length):
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