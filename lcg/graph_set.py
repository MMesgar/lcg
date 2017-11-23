###############################################################################
##                           Authour: Mohsen Mesgar
###############################################################################
import glob
import os
import numpy as np
import re
import sys, getopt
import networkx as nx
from subgraph_struc import draw
###############################################################################
##                           Line Patterns
###############################################################################
xpPtrn = re.compile("XP.*")
gNamePtrn = re.compile("% \d+")
vertexPtrn = re.compile("v .*")
edgePtrn = re.compile("d .*")
frequncyPtrn = re.compile("% => .*")


###############################################################################
##
###############################################################################
def read_graph_set(g_set_file, threshold=0.0):
    database_file = open(g_set_file)
    database_lines = database_file.readlines()
    database_file.close()
    #{id:{'name':...., 'graph':....}}
    graph_set = {}    
    
    gid = -1
    for line in database_lines[:] :
        if xpPtrn.match(line) :
            g=nx.DiGraph()
            gid = gid + 1
            graph_set[gid] = {'graph':g, 'name':''}
        elif vertexPtrn.match(line): # v (vertexId) (vertexlabel)
            vertexId= line.split()[1]		
            g.add_node(vertexId)
        elif edgePtrn.match(line): # d (start vertexID) (end vertexID) (edge label)
            startId= line.split()[1]
            endId= line.split()[2]
            if threshold>0.0:
                weight = float(line.split()[3])
                if weight > threshold: 
                    g.add_edge(startId, endId)
            else:
                g.add_edge(startId, endId)
        elif gNamePtrn.match(line):
            gname = line.split()[1]
            graph_set[gid]['name'] = gname
    return graph_set

###############################################################################
##                            write a graph_set
###############################################################################
def write_graph_set(gs_dic, path):
    output = ""
    for value in gs_dic.values():
        output += "XP\n"
        g = value['graph']
        name = value['name']
        output += ("% "+name+"\n")
        nodes = g.nodes()
        list.sort(nodes)
        for vertex in nodes:
            output += "v " + vertex +" a" +"\n"
        for edge in g.edges():
            output += "d "+ edge[0]+" "+ edge[1] + " 1.0\n"     
    with open(path, "w") as f:
        f.write(output)
    return output
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
##                              combine two graph dataset
###############################################################################
def combine(gs1_file, gs2_file):
    gs1 = read_graph_set(gs1_file)
    gs2 = read_graph_set(gs2_file)
    gs3 = {}
    assert(len(gs1)== len(gs2))
    
    for k,t in gs1.items():
        key = k 
        g1 = t['graph']
        name = t['name']
        g3 = nx.DiGraph()        
        e3 = []
        num_matched_graphs =0
        for v in gs2.values():
            if v['name']==name:
                num_matched_graphs += 1
                assert(num_matched_graphs==1)                
                g2 = v['graph']
                n3 = g1.nodes() + g2.nodes()
                e3 = g1.edges() + g2.edges()
                g3.add_nodes_from(n3)                
                g3.add_edges_from(e3)
                value = {'graph': g3 , 'name': name}
        assert(num_matched_graphs!=0)
        gs3[key] = value
    
    return gs3
        
###############################################################################
##                              Draw a Dataset
###############################################################################    
def draw_dataset(graph_set_dic):
    
    gs = graph_set_dic
    for t in gs.values():
        g = t['graph']
        name = t['name']
        print "name= "+ name
        draw(g)
        print "********************************************"
###############################################################################
##                              Main
###############################################################################
if __name__ == '__main__':
    eg_set= "/data/nlp/mesgarmn/Data/Readability_assessment/pitler08/ongoing/embedding/300/graphset/thershold/0.9/graph_set.g"
    emd_set = "/data/nlp/mesgarmn/Data/Readability_assessment/pitler08/ongoing/embedding/100/graphset/thershold/0.9/graph_set.g"
    #output = "/data/nlp/mesgarmn/Data/Readability_assessment/Orphee_RA_Data/ongoing/wprn_emd/graphset/thershold/0.5/graph_set.g"
    #com_set = combine(eg_set,emd_set)
    #op = write_graph_set(com_set, output )    
    #egs = read_graph_set(eg_set)
    #emds = read_graph_set(emd_set)
    
    n = "/data/nlp/mesgarmn/Data/Readability_assessment/Orphee_RA_Data/ongoing/embedding/300/output/threshold/0.9/gspan_s1.0_minn_4_maxn_4.induced.graphs"
    #n= "/data/nlp/mesgarmn/Data/Readability_assessment/Orphee_RA_Data/ongoing/w_pron/output/gspan_s1.0_minn_3_maxn_3.induced.graphs"
    g = read_graph_set(n)
    
    