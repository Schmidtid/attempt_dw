# %%
import numpy as np
import networkx as nx
#import dwave_networkx as dnx
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from matplotlib import pyplot as plt
from collections import defaultdict
n = 100  # number of nodes
G = nx.Graph()
h = defaultdict(int)
J = defaultdict(int)
f = open(r"/workspace/Attempt_optimizer/d-w/output_500.txt")
lines = f.readlines()
for line in lines:
    l = line.split()
    J[(int(l[0]),int(l[1]))] = int(l[2])
f.close() 
G.add_edges_from(J)
#for (u, v) in G.edges():
#    J[(u,v)] = 2*np.random.randint(-1,1)+1


# %%
numruns = 2000
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_ising(h, J,
                                num_reads=numruns,
                                label='Example - Maximum Cut Ising')
print('Set -1', 'Set 1', 'Energy', 'Cut Size')
for sample, E in response.data(fields=['sample','energy']):
    S0 = [k for k,v in sample.items() if v == -1]
    S1 = [k for k,v in sample.items() if v == 1]
    print(f'{str(S0)}, {str(S1)}, {str(E)}, {str(int((100-E)/2))}')
lut = response.first.sample
print(f"\nBest_spins_conf: {lut}\nInitial_edges: {J}",sep='\n')
print(f"{response.data(fields=['energy'])}\n")
# Interpret best result in terms of nodes and edges
S0 = [node for node in G.nodes if lut[node]==-1]
S1 = [node for node in G.nodes if lut[node]==1]
cut_edges = [(u, v) for u, v in G.edges if lut[u]!=lut[v]]
uncut_edges = [(u, v) for u, v in G.edges if lut[u]==lut[v]]

# Display best result
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, nodelist=S0, node_color='r')
nx.draw_networkx_nodes(G, pos, nodelist=S1, node_color='c')
nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=3)
nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=3)
plt.suptitle(f"Best cut:{str(int((100-E)/2))} with energy: {round(max(response.data(fields=['energy']))[0],3)}")
nx.draw_networkx_labels(G, pos)


filename = "04_100.png"
plt.savefig(filename, bbox_inches='tight', )
# %%
