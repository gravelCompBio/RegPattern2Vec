import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import networkx as nx
import math
import itertools

def split_process(data_path, edge_file, edge_of_interest, split_perc, train_file_name = 'df_train.csv', test_file_name='df_test.csv'):
    df_edges = load_edge_file(data_path + edge_file)
    
    candidates, MST = check_number_of_possible_cuts(df_edges, edge_of_interest)
    print('number of candidate edges to remove: {}'.format(len(candidates)))
    
    df_edge_of_interest = df_edges[df_edges['r'] == edge_of_interest]
    print('total number of edge of interest: {}'.format(df_edge_of_interest.shape))
    
    edge_to_index, index_to_edge = df_to_edge_to_index(df_edge_of_interest)
    print('edge_to_index: {}, index_to_edge: {}'.format(len(edge_to_index), len(index_to_edge)))
    
    candidates_indices = []
    for i in candidates:
        candidates_indices.append(edge_to_index[i])
    print('candidates_indices: {} == unique {}'.format(len(candidates_indices), len(set(candidates_indices))))
    
    split_number = min(int(split_perc * df_edge_of_interest.shape[0]), len(candidates)) 
    print('The minimum of {} and {} is {}.'.format(split_perc * df_edge_of_interest.shape[0], len(candidates), split_number ))
    
    edge_of_interest_indices_selected = random.sample(candidates_indices, k = split_number)
    df_test_edges = df_edge_of_interest.loc[edge_of_interest_indices_selected]
    print('selected indices: {}, df_test_edges: {}'.format(len(edge_of_interest_indices_selected), df_test_edges.shape)  )
    
    interest_rest = list(set(df_edge_of_interest.index) - set(edge_of_interest_indices_selected))
    df_interest_rest = df_edge_of_interest.loc[interest_rest]
    print('indices of the rest of interest: {}, df_interest_rest: {}'.format(len(interest_rest), df_interest_rest.shape))
    
    df_rest = df_edges[~df_edges.r.isin([edge_of_interest])]
    print('rest of edges except the edge of interest: {}'.format(df_rest.shape))
    
    df_train = pd.DataFrame([])
    df_train = df_train.append(df_interest_rest)
    df_train = df_train.append(df_rest)
    print('df_train: {}'.format(df_train.shape))
    print(df_train.shape[0] + df_test_edges.shape[0] == df_edges.shape[0])
    
    print('saving file {} ...'.format(test_file_name))
    save_file(df_test_edges, data_path, test_file_name)
    print('file {} and size {} is saved.'.format(test_file_name, df_test_edges.shape))
    
    print('saving file {} ...'.format(train_file_name))
    save_file(df_train, data_path, train_file_name)
    print('file {} and size {} is saved.'.format(train_file_name, df_train.shape))
    
    print('split is done.')
    
def load_edge_file(edge_file_path):
    df_edges = pd.read_csv(edge_file_path, dtype={'t':'str'})
    print('df_edges: {}'.format(df_edges.shape))
    return df_edges

def save_file(df, data_path, file_name):
    df.to_csv(data_path + file_name, index=False)

def df_to_edge_to_index(df):
    edge_to_index = dict()
    index_to_edge = dict()
    
    for i in df.itertuples():
        index_, h,t, r, h_id, t_id, r_id, h_c, t_c, h_c_id, t_c_id = i
        
        tup = (h_id, t_id, r_id)
        rev_tup = (t_id, h_id, r_id)
        
        edge_to_index[tup] = index_
        edge_to_index[rev_tup] = index_
        
        index_to_edge[index_] = tup
        index_to_edge[index_] = rev_tup
    return edge_to_index, index_to_edge

def check_number_of_possible_cuts(df_graph, relation):
    
    df_interest = df_graph[df_graph['r'] == relation]
    
    G = load_graph(df_interest)
    cc_g, connected_components_list_len = Connected_component(G)
    total_from_cc = np.sum(connected_components_list_len) # IT WAS OKAY
    print('number of total node in data based on the Networkx\'s Graph {}'.format(total_from_cc) )
    
    count_msp = 0
    sum_edges = 0
    
    remove_candidate = set()
    all_MST = set()
    
    for c in cc_g:
        df_c = df_interest[(df_interest.h_id.isin(c)) & (df_interest.t_id.isin(c))]

        g = load_graph(df_c)
        
        T = nx.minimum_spanning_tree(g)
        MST = { (i[0], i[1], i[2]['weight'])for i in sorted(T.edges(data=True))}
        all_MST.update(MST)
        
        g_len = len(g.edges) 
        T_len = len(T.edges)
        if g_len>T_len :
            sum_edges += g_len - T_len
            count_msp+=1
            
            g   = { (i[0], i[1], i[2]['weight'])for i in sorted(g.edges(data=True))}
            
    
            tmp = g - MST
            remove_candidate.update(tmp)
            
            
            
    print('df_interest: {}, count_msp: {}, sum_edges: {}'.format(df_interest.shape[0],count_msp, sum_edges))
    
    remove_nodes = set(itertools.chain(*remove_candidate)) 
    remain_nodes = set(itertools.chain(*all_MST)) 
    
    if len(remove_nodes - remain_nodes) == 0:
        print('all removed nodes {} in the train {}(should be equal to nodes in graph).'.format(len(remove_nodes), len(remain_nodes)))
    
    
    
    return remove_candidate, all_MST


def load_graph(df):
    df_edges = df[['h_id', 't_id','r_id']]
    edges = [tuple(x) for x in df_edges.values]

    GRAPH = nx.Graph()   
    GRAPH.add_weighted_edges_from(edges)
    return GRAPH

def Connected_component(G):
    connected_components_list_len = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    cc_g = list( nx.connected_components(G) )
    total_from_cc = np.sum(connected_components_list_len) # IT WAS OKAY

    return cc_g, connected_components_list_len