
import sys
sys.path.append("..") 

import pickle
import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from gnn_model import *
from utils import *
from scoring import mlp_score
import random
import networkx as nx
import matplotlib
import matplotlib.pyplot

log_print		= get_logger('testrun', 'log', get_config_dir())
def get_data_stats(data_name, data_type, dir_path, cold_perc, blind_type):
    if cold_perc == 1.0:
        path = dir_path+ '/{}/{}_graphs.pkl'.format(data_name, data_name)
    else:
        if cold_perc == 0.0:
            path = dir_path+ '/{}/{}_true_graphs.pkl'.format(data_name, data_name)
        else:
            perc_str = '25'
            if cold_perc == 0.75:
                perc_str = '75'

            path = dir_path+ '/{}/{}_{}_{}_graphs.pkl'.format(data_name, data_name, perc_str, blind_type)

    print("Opening data set", path)
    graphs_input = open(path, 'rb')
    graphs = pickle.load(graphs_input)


    min_edges = float('inf')
    max_edges = 0.0
    total_edges = 0.0
    min_nodes = float('inf')
    max_nodes = 0.0
    total_nodes = 0.0
    min_clustering = float('inf')
    max_clustering = 0.0
    total_clustering = 0.0
    min_neighbors = float('inf')
    max_neighbors = 0.0
    total_neighbors = 0.0
    min_isolates = float('inf')
    max_isolates = 0.0
    total_isolates = 0.0
    min_components = float('inf')
    max_components = 0.0
    total_components = 0.0
    #min_diameter = float('inf')
    #max_diameter = 0.0
    #total_diameter = 0.0
    total_graphs = 0.0

    print(data_name,"raw data")
    print("Nodes,edges,clustering coefficient,neighbors,isolated nodes,connected components")
    edge_type_counts = {0: 0,
                        1: 0,
                        2: 0}
    feature_edge_type_counts = {0: 0,
                        1: 0,
                        2: 0}
                        
    for graphs_key in graphs.keys():
        for graph in graphs[graphs_key]:
            if cold_perc == 1.0:
                edges = graph['pos_edges']
            else:
                edges = graph['given_edges']
                
            num_nodes = graph['node_count']
            total_graphs += 1

            if num_nodes > max_nodes:
                max_nodes = num_nodes
            if num_nodes < min_nodes:
                min_nodes = num_nodes
            total_nodes += num_nodes

            num_edges = len(edges)

            if num_edges > max_edges:
                max_edges = num_edges
            if num_edges < min_edges:
                min_edges = num_edges
            total_edges += num_edges

            adj = np.zeros((num_nodes, num_nodes))

            for edge in edges:
                adj[edge[0], edge[1]] = 1.0

            nx_graph = nx.from_numpy_matrix(adj)

            clustering = nx.average_clustering(nx_graph)

            #edge_feats = graph['edge_features']
                #print('Graph', str(total_graphs))

            #for feat in edge_feats:
            #    edge_t = int(torch.argmax(feat).item())
            #    feature_edge_type_counts[edge_t] += 1

            '''if clustering == 0.0 and nx_graph.number_of_edges() > 0:
                node_feats = graph['gnn_feature']
                #print('Graph', str(total_graphs))

                for edge_index in range(len(edges)):
                    line = ''
                    if node_feats[edges[edge_index][0]][3] > 0.0:
                        line += str(edges[edge_index][0]) + ', Blue'
                    else:
                        line += str(edges[edge_index][0]) + ', Red'

                    #edge_type = int(torch.argmax(edge_feats[edge_index]).item())

                    if edge_type == 0:
                        line += '---Interactable-->'
                    if edge_type == 1:
                        line += '---Attacking-->'
                    if edge_type == 2:
                        line += '---Supporting-->'

                    edge_type_counts[edge_type] += 1

                    if node_feats[edges[edge_index][1]][3] > 0.0:
                        line += str(edges[edge_index][1]) + ', Blue'
                    else:
                        line += str(edges[edge_index][1]) + ', Red' 

                    #print(line)

                for node in nx_graph.nodes:
                    if node_feats[node][3] > 0.0:
                        nx.set_node_attributes(nx_graph, {node: {'Team': 'Blue'}})
                    else:
                        nx.set_node_attributes(nx_graph, {node: {'Team': 'Red'}})

                for i in range(len(edges)):
                    edge_key = (edges[i][0], edges[i][1])

                    if edge_feats[i][0] > 0.0:
                        nx.set_edge_attributes(nx_graph, {edge_key: {'Type': 'Interactable'}})
                    if edge_feats[i][1] > 0.0:
                        nx.set_edge_attributes(nx_graph, {edge_key: {'Type': 'Attacking'}})
                    if edge_feats[i][2] > 0.0:
                        nx.set_edge_attributes(nx_graph, {edge_key: {'Type': 'Supporting'}})
                fig = matplotlib.pyplot.figure()
                pos = nx.spring_layout(nx_graph, seed=0)
                state_pos = {n: (x+0.12, y+0.05) for n, (x,y) in pos.items()}
                node_states = nx.get_node_attributes(nx_graph, 'Team')
                edge_states = nx.get_edge_attributes(nx_graph, 'Type')
                nx.draw_networkx(nx_graph, pos, node_size=600)
                nx.draw_networkx_labels(nx_graph, state_pos, labels=node_states, font_color='red')
                nx.draw_networkx_edge_labels(nx_graph, state_pos, edge_labels=edge_states)
                fig.savefig(str(total_graphs)+"_graph_plot.png")'''
                #print("Number of edges with clustering coefficient", nx_graph.number_of_edges())

            if clustering > max_clustering:
                max_clustering = clustering
            if clustering < min_clustering:
                min_clustering = clustering
            total_clustering += clustering

            total_node_neighbors = 0.0

            for i in range(num_nodes):
                total_node_neighbors += nx_graph.degree(i)

            neighbors = total_node_neighbors / num_nodes

            if neighbors > max_neighbors:
                max_neighbors = neighbors
            if neighbors < min_neighbors:
                min_neighbors = neighbors
            total_neighbors += neighbors

            isolates = nx.number_of_isolates(nx_graph)

            if isolates > max_isolates:
                max_isolates = isolates
            if neighbors < min_isolates:
                min_isolates = isolates
            total_isolates += isolates

            components = nx.number_connected_components(nx_graph)

            if components > max_components:
                max_components = components
            if components < min_components:
                min_components = components
            total_components += components

            '''isolated_nodes = nx.isolates(nx_graph)
            nx_graph = nx_graph.remove_nodes_from(list(isolated_nodes))
            diameter = nx.diameter(nx_graph)

            if diameter > max_diameter:
                max_diameter = diameter
            if diameter < min_diameter:
                min_diameter = diameter
            total_diameter += diameter'''


            print(num_nodes,",",num_edges,",",clustering,",",neighbors,",",isolates,",",components)

    print("Overal statistics out of", total_graphs, "graphs")
    print("Avg. Nodes, Avg. Edges, Avg. Clustering Coefficient,Avg. Neighbors,Avg. Isolated Nodes,Avg. Connected Components")
    print(total_nodes / total_graphs,",",total_edges / total_graphs,",",total_clustering / total_graphs,",", total_neighbors / total_graphs,",", total_isolates / total_graphs,",", total_components / total_graphs)
    print("Max Nodes, Max Edges, Max Clustering Coefficient,Max Neighbors,Max Isolated Nodes,Max Connected Components")
    print(max_nodes,",",max_edges,",",max_clustering,",", max_neighbors,",", max_isolates,",", max_components)
    print("Min Nodes, Min Edges, Min Clustering Coefficient,Min Neighbors,Min Isolated Nodes,Min Connected Components")
    print(min_nodes,",",min_edges,",",min_clustering,",", min_neighbors,",", min_isolates,",", min_components)

def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='reddit')
    parser.add_argument('--data_type', type=str, default='simple')
    parser.add_argument('--cold_perc', type=float, default=1.0)
    parser.add_argument('--blind_type', type=str, default='edge')
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))

    args = parser.parse_args()

    get_data_stats(args.data_name, args.data_type, args.input_dir, args.cold_perc, args.blind_type)


if __name__ == "__main__":
    main()
   