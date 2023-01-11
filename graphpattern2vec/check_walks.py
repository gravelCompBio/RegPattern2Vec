import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import numpy as np

def load_walsk(filepath):
    walks = []
    with open(filepath, 'r') as read_obj:

        reader = csv.reader(read_obj)
        for row in reader:
            lst = row[0].strip().split(' ')
            w = [int(i[1:]) for i in lst]
            walks.append(w)


    #     walks = list(csv_reader)
    print('walks: {}'.format(len(walks)))
    return walks

def convert_walks_to_edges(walks):
    edges = set()
    for i in walks:
        s = [e for e in i[1:]]
        for j in range(len(s) - 1):
            edges.add((s[j], s[j+1]))
    print('Edges: {}'.format(len(edges)))
    return edges

def check_walk_in_edge(lst, e_tup):
    c, t = 0, 0
    for i in tqdm(lst):
        
        head = i[0]
        tail = i[1]
        
        
        if ((head, tail) not in e_tup) and ((tail, head) not in e_tup):
            c+=1
#             print(head, tail)
        elif ((head, tail)  in e_tup) or ((tail, head)  in e_tup):
            t+=1
     
    print('not exist {} out of {}={}'.format(c, len(lst), t))
    
def get_metapaths(walks, G, d_types, id_to_type, source_type_name, target_type_name):
    mps = set()
    for w in walks:
#         print([int(i[1:]) for i in w])
        mp = tuple([d_types[id_to_type[i]] for i in w])
        mps.add(mp)
        
    err_c = 0    
    print('target_type_name[0]: {}'.format(target_type_name[0]))
    basic_mp = set()
    for mp in mps:
        lst = list(mp)

        while len(lst) > 0:

            try:
                
                i_pat = lst.index(target_type_name[0])
                basic_mp.add(tuple(lst[:i_pat+1]))
                lst = lst[i_pat+1:]
            except ValueError as e:
                err_c+=1
                basic_mp.add(tuple(lst))
                lst = []  
                
#     print(basic_mp)

    total_count, good_count = 0, 0

    for i in basic_mp:

        total_count+=1
        if (i[0] in source_type_name) and (i[-1] in target_type_name ):
#             print(i)
            good_count+=1
    print('total: {}, Good: {}, Err_c: {}'.format(total_count, good_count, err_c))
    
    return basic_mp

def generate_plots(walks,d_types, id_to_type):
    d_w = dict()
    for w in walks:
        for i in w:
            n_id = i
            if n_id not in d_w:
                d_w[n_id] = 0
            d_w[n_id] +=1
            
    d_type_freq  = dict()
    for n in d_w:
        node_freq = d_w[n]
        type_id = id_to_type[n]

        t_n_G = int(id_to_type[n])
        if t_n_G!=type_id:
            print(n, type_id, t_n_G)
            print('Error that should not occur')
        type_name = d_types[type_id]
        if type_name not in d_type_freq:
            d_type_freq[type_name] = 0
        d_type_freq[type_name] += node_freq

    arr_freq = np.array([[i, d_type_freq[i]] for i in d_type_freq])
    
    d_type_count = dict()
    for n in d_w:
        node_freq = d_w[n]
        type_id = id_to_type[n]
        type_name = d_types[type_id]

        if type_name not in d_type_count:
            d_type_count[type_name] = 0
        d_type_count[type_name] += 1

    arr_count = np.array([[i, d_type_count[i]] for i in d_type_count])
#     print('Frequent:')
    avg_appearance_of_types_histogram(arr_freq, 'Frequency of types')
#     print('count:')
    avg_appearance_of_types_histogram(arr_count, 'Unique number of each types')

    d_type_avg = dict()
    for t in d_types:
        type_name = d_types[t]
        if type_name in d_type_freq:
            t_count = d_type_count[type_name]
            t_freq = d_type_freq[type_name]
            t_avg = t_freq / t_count
            d_type_avg[type_name] = t_avg

    arr_avg = np.array([[i, d_type_avg[i]] for i in d_type_avg])
#     print('average:')
    avg_appearance_of_types_histogram(arr_avg, 'Average Frequency of types')

def avg_appearance_of_types_histogram(lst, title = 'No Title'):
    arr = np.array(lst)
    
    x = np.arange(len(arr[:, 0]))
    labels = arr[:,0]
#     labels = [d_types[i] for i in x]
    y = np.array([float(i) for i in arr[:, 1]])
    
    fig, ax = plt.subplots(figsize=(10,10))
    plt.bar(x=x, height=y)
    
    fig.suptitle(title, fontsize=16)
    plt.xticks(x, labels, rotation='vertical')
#     plt.xticks(x, rotation='vertical')

def draw_plot(d):
    x = []
    x_labels= []
    for i in d:
        x_labels.append(i)
        x.append(G.degree[i])
    y = list(d.values())
    plt.scatter(x, y)  
    
def draw_plot_labels(d):
    x = []
    x_labels= []
    for i in d:
        x_labels.append(i)
        x.append(G.degree[i])
    y = list(d.values())
             
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(x, y)

    for i, txt in enumerate(x_labels):
        ax.annotate(txt, (x[i], y[i]))
        
def load_graph(df):
    
    print('load_graph| df : {}'.format(df.shape))
    df_edges = df[['h_id', 't_id','r_id']]
    edges = [tuple(x) for x in df_edges.values]
    print('load_graph| edges: {}'.format(len(edges)))

    GRAPH = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
    GRAPH.add_weighted_edges_from(edges)
    return GRAPH

def get_dict_types(df_train):
    h_types = dict(tuple(zip(df_train.h_c_id,df_train.h_c)))
    d_types = dict(tuple(zip(df_train.t_c_id,df_train.t_c)))
    d_types.update(h_types)
#     print(d_types)
    
    h = dict(tuple(zip(df_train.h_id,df_train.h_c_id)))
    id_to_type = dict(tuple(zip(df_train.t_id,df_train.t_c_id)))
    id_to_type.update(h)
    
#     print(id_to_type)
    
    return d_types, id_to_type

def check_walks(data_path,  walk_file,train_file, source_type_name, target_type_name, chk_mp = False, chk_hist=False, draw_plot = False):
    df_train = pd.read_csv(data_path + train_file, dtype={'r':'str','h':'str', 't':'str'})
    print('df_train: {}'.format(df_train.shape))
    
    e_tup = {(i[4],i[5]) for i in df_train.itertuples()}
    print('e_tup: {}, df_train.shape[0]: {}'.format(len(e_tup), df_train.shape[0]))
    
    walks = load_walsk(data_path+ walk_file)
    e_list = convert_walks_to_edges(walks)
    print('e_list: {}'.format(len(e_list)))
    check_walk_in_edge(e_list, e_tup)
    
    mp = None
    
    if chk_mp == True:
        G = load_graph(df_train)
        d_types, id_to_type = get_dict_types(df_train)
        mp = get_metapaths(walks, G, d_types, id_to_type, source_type_name, target_type_name)
    
    if chk_hist == True:
        len_of_walks =[len(w) for w in walks]
        _ = plt.hist(len_of_walks)
        
    if draw_plot == True:
        d_types, id_to_type = get_dict_types(df_train)
        generate_plots(walks, d_types, id_to_type)
        
    return walks, mp