import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tabulate import tabulate
import scipy.stats as sps

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


''' 
Currently uncritical but needs to be developed at some point iA
'''

def optimize_scale(precision = 0.01):
    best_scale_for_network = np.arange(1.5, 3.0, precision)
    # some extra steps
    return best_scale_for_network
    
    
'''
Current task: sample from distributions
    1. degree dist
    2. weight dist
'''

class degree_distribution(sps.rv_continuous): 
    def _pdf(self, x, a, b, c):
        return degree_func(x, a, b, c)
    
    def _argcheck(self, a, b, c):
        return True
    
class weight_distribution(sps.rv_continuous):
    def _pdf(self, x, a, b):
        return weight_func(x, a, b)
    
    def _argcheck(self, a, b):
        return True
    