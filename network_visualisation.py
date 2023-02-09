'''
Collection of functions to visualise propagation of contagion
'''

import os

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
        
def collect_frames(propagation_scheme, propagation_params, layout = 'spring'):
    files = glob.glob('/Users/ali/MSci Project/IF visualisation/frame*.png')
    for f in files:
        os.remove(f)
    
    
    
    return 0

def make_gif(dur):
    frames = []
    imgs = glob.glob('/Users/ali/MSci Project/IF visualisation/frame*.png')
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
        
    frames[0].save('/Users/ali/MSci Project/IF visualisation.gif', format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=dur)