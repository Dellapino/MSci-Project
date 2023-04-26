import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tabulate import tabulate
import scipy.stats as sps
import networkx as nx

def analyse_graph(graph):
    weights = []
    degrees = []
    for node in graph.nodes:
        connections = graph.neighbors(node)
        degrees.append(graph.degree[node])
        weight = []
        for connection in connections:
            weight.append(graph[node][connection]['weight'])
        weights += weight
    return degrees, weights

def linear_binning(data, num_bins = 100, norm = 1): 
    '''
    finds midpoints and frequencies of data after binning, removes any zeros
    present in the data
    
    parameters: weights:    array
                            contains all weights present in a graph
    
    returns:    midpoints:  array
                            midpoints of histogram bins
                freqs:      array
                            frequecies of histogram bins
    '''                     
    freqs, edges = np.histogram(data, bins = num_bins, density = norm)
    midpoints = edges[:-1] + np.diff(edges)/2    
    to_remove = np.where(freqs == 0)[0]
    midpoints = np.delete(midpoints, to_remove)
    freqs = np.delete(freqs, to_remove)
    return midpoints, freqs

def log_binning(data, scale = 1.2, normed = True):
    if scale <= 1:
        raise ValueError('Function requires scale > 1')
    
    if normed:
        start = min(data)
        count = 0 # keeps track of how many orders of magnitude we need to increase to get start to be greater than 1
        while start < 1:
            start = start * 10
            count += 1
        end  = 10 ** count
        bin_edges = [0, start]
        while bin_edges[-1] < end:
            bin_edges.append(bin_edges[-1] ** scale)
        bin_edges = np.array(bin_edges) / end
    
    else:
        end = max(data)
        wmax = np.ceil(np.log(end) / np.log(scale))
        bin_edges = scale ** np.arange(1,wmax + 1)
        
    freqs, edges = np.histogram(data, bins = bin_edges, density = 0)
    midpoints = edges[:-1] + np.diff(edges)/2    
    
    to_remove = np.where(freqs == 0)[0]
    midpoints = np.delete(midpoints, to_remove)
    freqs = np.delete(freqs, to_remove)
    
    return midpoints[0:], freqs[0:]

def linear(x, m, c):
    return m*x + c

def reciprocal(x, a, b, c):
    return a/(x**b) + c

def exponential(x, a, b):
    return b*np.exp(x*a)

def logarithm(x, a, b):
    return a * np.log(x) + b

def power(x, a, b):
    return a * ((x)**b)

def weight_func(x, a, b):
    return a * ((x)**b)

def weight_func(x, a, b, c):
    return  b * ((x)**c) / np.exp(x*a)

def lambert_w_function():
    return 0

def log_normal(x, avg, std, a, b):
    return (1/((x+b)/a)*std) * np.exp(-np.log(((x+b)/a)-avg)**2/(2*std**2))

def maxwell_boltzmann(x, a, b, c, d):
    return d * (((x+b)/c)**2) * np.exp((-((x+b)/c)**2)/(2*a**2)) / (a**3) 

def poisson(x, avg): # x is not always integer could lead to problems
    return avg**x * np.exp(-avg) / np.math.factorial(x)

def degree_func(x, a, b, c): # this is amazing, got idea from https://harry45.github.io/blog/2016/10/Sampling-From-Any-Distribution
    return (x**a) / np.exp(b*x+c)

def check_fit(fit_func, bin_func, data, params, error = False, plot = False, plot_scale = 'linear'): 
    if plot:
        plt.figure()
    
    if error:
        x, y, e = bin_func(data, *params)
        fit, cov = curve_fit(fit_func, x, y, sigma = e, absolute_sigma = 1)
        if plot:
            plt.errorbar(x, y, yerr = e, fmt = 'x')
    else:
        x, y = bin_func(data, *params)
        fit, cov = curve_fit(fit_func, x, y)
        if plot:
            plt.errorbar(x, y, fmt = 'x')
        
    xx = np.linspace(min(x), max(x), 10000)
    
    if plot:
        plt.plot(xx, fit_func(xx, *fit))
        plt.yscale(plot_scale)
        plt.xscale(plot_scale)
        plt.show()
    
    err = np.sqrt(np.diag(cov))
    perr = abs(err * 100 / np.array(fit))
    
    return fit, perr

''' 
This is an uncritical function at the curent moment [INCOMPLETE] 
Also, check_fit contains this functionality perhaps at the cost of efficiency
'''
def plot_fit(fit_func, fit_params, plot_scale = 'log'):
     # this is going to break, need data x, y, e somehow without replicating check_fit()
    
    x = np.linspace(min(x), max(x), 10000)
    plt.figure()
    plt.plot(x, fit_func(x, *fit_params))
    plt.yscale(plot_scale)
    plt.xscale(plot_scale)
    plt.show()
    

def compare_scale(fit_func, data, scales, normed, show_all = False):  
    avg_errs = []
    for scale in scales:
        _, perr = check_fit(fit_func, log_binning, data, [scale, normed], plot_scale='log')
        avg_err = sum(perr) / len(perr)
        avg_errs.append(avg_err)

    best = avg_errs.index(min(avg_errs))
    if show_all:
        tabulated_data = [[scales[i], round(avg_errs[i], 2)] for i in range(len(scales))]
        print(tabulate(tabulated_data, headers = ['scaling factor', 'average error']))
    else:
        print(tabulate([[scales[best], round(avg_errs[best], 2)]], headers=['best scaling factor', 'best average error']))
    return scales[best], avg_errs[best] # pass these into optimize_scale

def optimize_scale(weights, precision = 0.01):
    scales = np.arange(1.5, 3.0, precision)
    best_scale = compare_scale(weight_func, weights, scales, False, False)
    return best_scale
    
'''
Current task: sample from distributions
    1. degree dist - Done
    2. weight dist - Done
'''
    
def inverse_weight_func(x, a, b):
    return (x/a) ** (1/b)

def fit_weights(weights, scale):
    x, y = log_binning(weights, scale, False)
    fit, cov = curve_fit(weight_func, x, y)
    return fit, cov

def fit_degrees(degrees):
    x, y = linear_binning(degrees, 8, 1)
    fit, cov = curve_fit(degree_func, x, y)
    return fit, cov

def sample_weights(sample_num, fit): # can add normalization if needed but I'm begining to see this as uneccessary
    randys = np.random.randint(1, 1000, size = sample_num)
    weight_samples = inverse_weight_func(randys, *fit)
    #norm = max(weight_samples)
    #weight_samples = weight_samples / norm
    return weight_samples

def set_weights(G, fit):
    nodes = list(G.nodes)
    for node in nodes:
        connections = list(G.neighbors(node))
        num_edges = len(connections)
        if num_edges == 0:
            print('What the dog doin')
        else:
            weights = sample_weights(len(connections), fit)
        for i, connection in enumerate(connections):
            G[node][connection]['weight'] = weights[i]
    return G   

def set_weights(G, sample_func):
    nodes = list(G.nodes)
    for node in nodes:
        connections = list(G.neighbors(node))
        num_edges = len(connections)
        if num_edges == 0:
            print('What the dog doin')
        else:
            weights = sample_func.rvs(size = len(connections))
            weights = np.ceil(weights).astype(int) # will only work in non normed case
            weights = [w for w in weights if w !=0]
        for i, connection in enumerate(connections):
            G[node][connection]['weight'] = weights[i]
    return G   

def generate_graph(num_nodes, dfit, wfit):
    degree_dist = degree_distribution(a = 0) # need to use dfit here somehow
    degree_samples = degree_dist.rvs(size = num_nodes)  
    degree_samples = np.ceil(degree_samples).astype(int)  
    G = nx.configuration_model(degree_samples, create_using = nx.Graph)
    G = set_weights(G, wfit)
    return G

'''
Current task:
    1. Find average weight (multiple schemes to define what is considered 1 interaction)
    2. Use this to inform threshold + see how many contacts before we expect infection
    3. Check if larger and smaller graphs have same properties
    4. Implement Neuro-SIR model
'''

def average_interaction_1(weights): # Basic version
    return sum(weights) / len(weights)

def average_interaction_2(fit): # via setting derivate of invse function == -1 for graph transition
    a = fit[0]
    b = fit[1]
    return a * ((-b) ** (b/(1-b)))

def average_interaction_3(fit):
    a = fit[0]
    b = fit[1]
    gradient = -1
    return (gradient/(a*b)) ** (1/(b-1))

def threshold(mins): # threshold based on sociopatterns data being in 20 sec increments
    return 3*mins

def degree_centrality(G):
    connectivty = nx.degree_centrality(G)
    nodes = list(connectivty.keys())
    total = 0
    for node in nodes:
        total += connectivty[node]
    return total

def degree_centrality(G):
    nodes = G.number_of_nodes()
    edges = G.number_of_edges() * 2
    return edges / nodes

def weight_centrality(G):
    nodes = list(G.nodes)
    total = 0
    for node in nodes:
        connections = G.neighbors(node)
        subtotal = 0
        for connection in connections:
            subtotal += G[node][connection]['weight']
        total += subtotal
    return total / G.number_of_edges()

def weight_centrality(G):
    nodes = list(G.nodes)
    total = 0
    for node in nodes:
        connections = G.neighbors(node)
        subtotal = 0
        for connection in connections:
            subtotal += G[node][connection]['weight']
        total += subtotal
    return total / (G.number_of_edges() * 2)

def closeness_centrality(G):
    connectivty = nx.closeness_centrality(G)
    nodes = list(connectivty.keys())
    total = 0
    for node in nodes:
        total += connectivty[node]
    return total

def betweenness_centrality(G):
    connectivty = nx.betweenness_centrality(G)
    nodes = list(connectivty.keys())
    total = 0
    for node in nodes:
        total += connectivty[node]
    return total

def eigenvector_centrality(G):
    connectivty = nx.eigenvector_centrality(G)
    nodes = list(connectivty.keys())
    total = 0
    for node in nodes:
        total += connectivty[node]
    return total

def measure_nodes(G): # These are measures are the node level
    # measures are based on https://dshizuka.github.io/networkanalysis/04_measuring.html#components
    centrality_measures = {}
    centrality_measures['degree'] = degree_centrality(G)
    centrality_measures['weight'] = weight_centrality(G)
    centrality_measures['closeness'] = closeness_centrality(G)
    centrality_measures['betweenness'] = betweenness_centrality(G)
    centrality_measures['eigenvector'] = eigenvector_centrality(G)
    return centrality_measures

def measure_network(G):
    # measures are based on https://dshizuka.github.io/networkanalysis/04_measuring.html#components
    network_measures = {}
    network_measures['nodes'] = G.number_of_nodes()
    network_measures['edges'] = G.number_of_edges()
    network_measures['ratio'] = network_measures['edges'] / network_measures['nodes']
    network_measures['components'] = nx.number_connected_components(G)
    network_measures['density'] = nx.density(G)
    network_measures['path'] = nx.average_shortest_path_length(G)
    network_measures['diameter'] = nx.diameter(G)
    network_measures['transitivity'] = nx.transitivity(G) # basically global clustering
    return network_measures

def print_dict(dict):
    for key in list(dict.keys()):
        print(key + ' : ' + str(dict[key]))
    print('\n')