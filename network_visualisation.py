'''
Collection of functions to visualise propagation of contagion
'''

import os
import SI_model_nx as sinx
import networkx as nx
import matplotlib.pyplot as plt
import glob
from PIL import Image

def plot(N, P, T, layout = 'spring'):
    files = glob.glob('/Users/ali/MSci Project/IF visualisation/frame*.png')
    for f in files:
        os.remove(f)

    H = nx.Graph(N)

    if layout == 'spring':
        positions = nx.spring_layout(H)
    elif layout == 'kamada':
        positions = nx.kamada_kawai_layout(H)
    else:
        positions = nx.kamada_kawai_layout(H)
        
    plt.figure()
    for t in range(T):
        P, F = propagate(P, N)
        colours = []
        for i in range(len(H)):
            if F[i] == 1:
                colours.append('yellow')
            else:
                colours.append('grey')
        plt.clf()
        #time.sleep(0.5)
        nx.draw_networkx(H, pos = positions, with_labels = False, node_size = 50, node_color = colours, alpha = 0.5)
        plt.savefig('/Users/ali/MSci Project/IF visualisation/frame' + str(t))
        
        
def plot_SI(N, P, T, layout = 'spring'):
    files = glob.glob('/Users/ali/MSci Project/IF visualisation/frame*.png')
    for f in files:
        os.remove(f)

    H = nx.Graph(N)

    if layout == 'spring':
        positions = nx.spring_layout(H)
    elif layout == 'kamada':
        positions = nx.kamada_kawai_layout(H)
    else:
        positions = nx.kamada_kawai_layout(H)
        
    plt.figure()
    for t in range(T):
        P, F = propagate_SI(P, N)
        colours = []
        for i in range(len(H)):
            if F[i] == 1:
                colours.append('yellow')
            else:
                colours.append('grey')
        plt.clf()
        #time.sleep(0.5)
        nx.draw_networkx(H, pos = positions, with_labels = False, node_size = 50, node_color = colours, alpha = 0.5)
        plt.savefig('/Users/ali/MSci Project/IF visualisation/frame' + str(t))        
    
def collect_frames_nx(G, initial, threshold, T, layout = 'spring'):
    files = glob.glob('/Users/ali/MSci Project/IF visualisation/frame*.png')
    for f in files:
        os.remove(f)
        
    G = sinx.initialise_potential(G, initial, threshold)
    
    
    positions = nx.spring_layout(G)
    #positions = nx.kamada_kawai_layout(G)
    #positions = nx.nx_agraph.graphviz_layout(G)
    plt.figure(figsize = (12, 8))
    for t in range(T):
        G = sinx.propagate(G, threshold)
        colors = []
        for node in list(G.nodes):
            if G.nodes[node]['potential'] >= threshold:
                colors.append('red')
            else:
                colors.append('grey')
        nx.draw_networkx(G, pos = positions, with_labels = 0, node_size = 100, node_color = colors, alpha = 0.5)
        plt.savefig('/Users/ali/MSci Project/IF visualisation/frame' + str(t))
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
                duration=200) # duration of each frame in milliseconds!
    return frames