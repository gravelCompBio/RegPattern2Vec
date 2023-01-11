import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import networkx as nx
import math


test_file = 'df_test.csv'
train_file = 'df_train.csv'


def read_edges(data_path, filename):
#     filename = 'df_edge2.csv'
    dtype = {'h': 'str', 'r': 'str', 't': 'str', 'h_c': 'str', 't_c':'str'}
    df = pd.read_csv(data_path + filename, dtype=dtype)  
    
    h_c_list = set(tuple(zip(df.h_c_id, df.h_c)))
    t_c_list = set(tuple(zip(df.t_c_id, df.t_c)))
    c_list = list(set( list(h_c_list) + list(t_c_list) ))
    df_type = pd.DataFrame(c_list, columns=['id', 'name'])
    
    h_nodes = set(tuple(zip(df.h_id, df.h, df.h_c_id)))
    t_nodes = set(tuple(zip(df.t_id, df.t, df.t_c_id)))
    nodes = list(set( list(h_nodes) + list(t_nodes) ))
    df_node = pd.DataFrame(nodes, columns=['id', 'name', 'type_id'])
    

    relation_list = set(tuple(zip(df.r_id, df.r)))
    df_relation = pd.DataFrame(relation_list, columns=['id','name'])
    
    return df, df_node, df_relation, df_type

def randomly_sample_df(df, rnd_state = 1, split_ratio = 3, check_for_counts = True):
    split_number = int(df.shape[0] / split_ratio)
    
    df_short = df[['h_id', 't_id', 'r_id']]
    d = df_to_dict(df_short)
    d_count = df_to_count(df_short)
    
    idx_list = df_short.index
    idx = random.choices(idx_list, k = split_number)
    
    selected_idx = list()
    
    for i in tqdm(range(len(idx))):
        
        res = df_short.iloc[i, [0, 1]].to_numpy()
        h_id = res[0]
        t_id = res[1]
        
        if check_for_counts == True:
            if (len(d[h_id]) > math.ceil(d_count[h_id]/2)) and (len(d[t_id]) > math.ceil(d_count[t_id]/2)) :
                try:
                    d[h_id].remove(t_id)
                    d[t_id].remove(h_id)
                    selected_idx.append(i)
                except KeyError:
                    print(idx, h_id, t_id, d[h_id], d[t_id])
        else:
            try:
                d[h_id].remove(t_id)
                d[t_id].remove(h_id)
                selected_idx.append(i)
            except KeyError:
                print(idx, h_id, t_id, d[h_id], d[t_id])
            


    print('randomly_sample_df| delete dictionary.')        
    del(d)
    
    print('randomly_sample_df| selected_idx: {}, split_number: {}'.format(len(selected_idx), split_number))
    
    df_sample = df.iloc[selected_idx]
    
    return df_sample

def df_to_count(df):
    d = dict()
    for i in df.itertuples():
        
        h_id = i[1]
        t_id = i[2]
        
        if h_id not in d:
            d[h_id] = 0
        d[h_id]+=1
        
        if t_id not in d:
            d[t_id] = 0
        d[t_id]+=1
    print('df_to_dict| d:{}'.format(len(d)))
    return d

def df_to_dict(df):
    d = dict()
    for i in df.itertuples():
        
        h_id = i[1]
        t_id = i[2]
        
        if h_id not in d:
            d[h_id] = set()
        d[h_id].add(t_id)
        
        if t_id not in d:
            d[t_id] = set()
        d[t_id].add(h_id)
    print('df_to_dict| d:{}'.format(len(d)))
    return d

def split_df(df,df_sample):
    sample_index = df_sample.index
    df_rest = df.loc[~df.index.isin(sample_index)]
    print('split_df| df_rest: {}, df_sample: {}, df: {}'.format(df_rest.shape, df_sample.shape, df.shape))
    return df_rest

def dict_node_to_class(df_e):
    node_class = dict()
    for i in df_e.itertuples():
        h = i[1]
        t = i[2]
        h_c = i[-4]
        t_c = i[-3]

        node_class[h] = h_c
        node_class[t] = t_c
        
    return node_class

def split_process(edge_file, rel, data_path = 'data/', check_for_counts = True, NoSplit = False, rnd_state = 1, split_ratio = 3):
    df_edges, df_node, df_relation, df_type = read_edges(data_path, edge_file)
    print('df_edges: {}, df_node: {}, df_relation: {}, df_type: {}'.format(df_edges.shape, df_node.shape, df_relation.shape,
                                                             df_type.shape))
    df_edges = df_edges[['h', 't', 'r', 'h_id', 't_id', 'r_id', 'h_c', 't_c', 'h_c_id', 't_c_id' ]]
    
    
    df_edges.to_csv(data_path + 'df_edges.csv', index=False)
    print('df_edges saved!')
    
    node_to_class = dict_node_to_class(df_edges)
    list_n_type = [node_to_class[i] for i in list(df_node['name'])]
    df_node = df_node.assign(type = list_n_type)
    df_node.to_csv(data_path + 'df_node.csv', index=False)
    print('df_node saved!')
    
    df_relation.to_csv(data_path + 'df_relation.csv', index=False)
    print('df_relation saved!')
    
    print(df_type)
    
    df_type.to_csv(data_path + 'df_type.csv', index=False)
    print('df_type saved!')
    
    if NoSplit == False:
    
        df_interest = df_edges[df_edges['r'] == rel]
        df_interest_sampled = randomly_sample_df(df_interest, rnd_state = rnd_state, split_ratio = split_ratio, check_for_counts = check_for_counts)

        df_rest = split_df(df_edges, df_interest_sampled)

        print('split_process| saving...')

        df_interest_sampled.to_csv(data_path + test_file, index=False)
        df_rest.to_csv(data_path + train_file, index=False)
        print('split_process| saving done.')

        print('df_edges: {}, df_train: {}, df_test: {}'.format(df_edges.shape, df_rest.shape, df_interest_sampled.shape))
    
        G = load_graph(df_rest)
        cc_g, connected_components_list_len = Connected_component(G)    
    
        return df_edges, df_rest, df_interest_sampled, cc_g, connected_components_list_len
    
    return df_edges

def load_graph(df):
    
    print('load_graph| df : {}'.format(df.shape))
    df_edges = df[['h_id', 't_id','r_id']]
    edges = [tuple(x) for x in df_edges.values]
    print('load_graph| edges: {}'.format(len(edges)))

    GRAPH = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
    GRAPH.add_weighted_edges_from(edges)
    return GRAPH

def Connected_component(G):
    connected_components_list_len = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    print('__check_connectivity| number of connected components: {}'.format(len(connected_components_list_len)))
    cc_g = list( nx.connected_components(G) )
    total_from_cc = np.sum(connected_components_list_len) # IT WAS OKAY
    print('__check_connectivity| number of total node in data based on the Networkx\'s Graph {}'.format(total_from_cc) )
    return cc_g, connected_components_list_len