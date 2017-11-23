###############################################################################
##                  author: Mohsen Mesgar
###############################################################################
import numpy as np
import multiprocessing as mp
import networkx as nx
from gensim.models import Word2Vec
from itertools import chain, combinations
from collections import defaultdict
import sys, copy, time, math, pickle
import itertools
import scipy.io
import pynauty
from scipy.spatial.distance import pdist, squareform
import glob
import os
import re
import sys, getopt
import scipy as sp
import matplotlib.pyplot as plt
import codecs
###############################################################################
##                      get adjacency matrix
###############################################################################
def graph_to_adj_matrix(graph):
    n = len(graph.nodes())
    am = np.zeros((n,n))
    for u,v in graph.edges():
        am[int(u),int(v)] = 1
    return am
###############################################################################
##             construct all graphs with k nodes
###############################################################################
def make_graphs(k):
    graphs = []
    adj = np.zeros((k,k))
    for i in range(k):
        for j in range(i+1,k):
            adj[i,j] =1   
    g = nx.DiGraph(adj)
    
    n = (k*k-k)/2
    stuff = g.edges()
    for l in range(0, len(stuff)+1):
        for subset in itertools.combinations(stuff,l):
            tmp = g.copy()
            for i,j in subset:
                tmp.remove_edge(i,j)
            graphs.append(tmp)
    return graphs

###############################################################################
##                       canonical map of a graph
###############################################################################
def get_canonical_map(g):
    if len(g.nodes())>0:
        a = nx.adjacency_matrix(g)
        am = a.todense()
        window = np.array(am)
        adj_mat = {idx: [i for i in list(np.where(edge)[0]) if i!=idx] for idx, edge in enumerate(window)}
#       This line doesn't take into account the order of nodes, it produce the identical
#       canonoical map for these graphs
#       0-->1 2, 0 1-->2, 0-->2 1
#        tmp = pynauty.Graph(number_of_vertices=len(g.nodes()), directed=True, adjacency_dict = adj_mat) 

        tmp = pynauty.Graph(number_of_vertices=len(g.nodes()), 
            directed=True, 
            adjacency_dict = adj_mat, 
            vertex_coloring = [set([t]) for t in range(len(g.nodes(0)))]) 

        cert = pynauty.certificate(tmp)
    else:
        cert = ''
    return cert
    
###############################################################################
##                       serialize a graph set
###############################################################################
def encode_graph(g, index):
    can_map = get_canonical_map(g)
    n = len(g.nodes())
    
    if n>0:
        adj_mat = nx.adj_matrix(g).toarray()
        graph = tuple(adj_mat[np.triu_indices(n,1)])
    else:
        graph = tuple()
    
       
    idx = index
    key = can_map
    value = {'graph':graph, 'idx': idx, 'n':n}
    return key, value

###############################################################################
##                          encode graph set
## graphset is a list of nx_graph_object
###############################################################################
def encode_graph_set(graphset):
    inx = 0    
    output = dict()
    for g in graphset:
        k,v = encode_graph(g, inx)
        output[k]=v
        inx = inx+1
    return output      
###############################################################################
##                 make graph set of graph with size<max_k
###############################################################################
def make_graph_set(max_k):
    gs = []
    for k in range(max_k+1):
        graphs = make_graphs(k)
        for g in graphs:
            gs.append(g)
        print "graphs with "+str(k)+" nodes are created"
    return gs

###############################################################################
##                            write a dict in a file
###############################################################################
def write(d, path):
    with open(path, "wb") as f:
        pickle.dump(d, f)
###############################################################################
##                           read a dict from a file
###############################################################################
def read(path):
    with open(path, "rb") as f:
        return pickle.load(f)

###############################################################################
##                          recover a graph
###############################################################################
def recover_graph(adj_triu, n, idx):    
    adj = np.zeros((n,n))
    k = 0
    for i in range(n):
        for j in range(i+1,n):
            adj[i,j] = adj_triu[k]
            k = k + 1
    g = nx.DiGraph(adj, index = idx)
    return g.copy()
###############################################################################
##                               Draw a graph
###############################################################################
def draw(g):
    nx.draw_circular(g,node_size=1000, with_labels=True)
    plt.figure(0,figsize=(12,12))
    plt.show()

###############################################################################
##                  construct_hierarchical_relation(dic, max_d)
## dic: is a dictionary which represents {can_map:{'graph':(), 'idx':0, 'n':0}}
###############################################################################
def construct_hierarchical_relation(dic, max_d):
    himap = dict()    
    for d in range(1,max_d+1):
        childs_size_d = {k:v for k,v in dic.items() if v['n']==d}
        parents_size_d_1= {k:v for k,v in dic.items() if v['n']==d-1}
        for child in childs_size_d.values():
            child_graph = recover_graph(child['graph'],child['n'],child['idx'])
            child_inx = child['idx']
            for node in child_graph.nodes():
                child_cp = child_graph.copy()
                child_cp.remove_node(node)
                child_cm = get_canonical_map(child_cp)
                parent_inx = parents_size_d_1[child_cm]['idx']
                if himap.has_key(parent_inx):
                    if(himap.get(parent_inx).has_key(child_inx)):
                        himap.get(parent_inx)[child_inx]= himap.get(parent_inx)[child_inx] +1
                    else:
                        himap.get(parent_inx)[child_inx]= 1
                else:
                    himap[parent_inx] = {child_inx:1}
    return himap
    

###############################################################################
##                               Main
###############################################################################
if __name__ == "__main__":
    max_k = 6
    print "start making all graph with "+str(max_k)+" nodes"
    graphs = make_graph_set(max_k)
    print "all graphs are made."
    
    print "start decoding graphs..."
    dic = encode_graph_set(graphs)
    
    if os.path.exists("./canonical_map/can_map_maxk"+str(max_k)+".p"):
        os.remove("./canonical_map/can_map_maxk"+str(max_k)+".p") 
    write(dic, "./canonical_map/can_map_maxk"+str(max_k)+".p")


    himap = construct_hierarchical_relation(dic, max_k) 
    if os.path.exists("./canonical_map/himap_maxk"+str(max_k)+".p"):
        os.remove("./canonical_map/himap_maxk"+str(max_k)+".p")
    write(himap, "./canonical_map/himap_maxk"+str(max_k)+".p")

    print "Done!"
    