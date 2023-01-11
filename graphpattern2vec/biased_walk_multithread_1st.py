import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np
import time
import csv
import collections
import math
import gc
import sys

from multiprocessing import cpu_count
import multiprocessing




class BiasedWalk(object):

    data_path = 'data/'
    edge_file = 'df_edges.csv'
    train_file = 'df_train.csv'
    
    def __init__(self, graph_pattern, loop_index, type_loop,  data_path = 'data/',  train_file = 'df_train.csv', postfix = '', degree_biased = True, num_thread = 10):

        self.data_path = data_path
        
        self.train_file = train_file
        self.graph_pattern = graph_pattern
        self.loop_index = loop_index
        self.type_loop = type_loop
        self.postfix = postfix
        self.num_thread = num_thread
        
        print('loading data ...')
        self.load_data()
        
        self.degree_biased = degree_biased
        
#         print('self.nodes: {}'.format(sys.getsizeof(self.nodes)))
#         print('self.edges: {}'.format(sys.getsizeof(self.edges)))
#         print('self.type_to_node: {}'.format(sys.getsizeof(self.type_to_node)))
#         print('self.id_to_type: {}'.format(sys.getsizeof(self.id_to_type)))
#         print('self: {}'.format(sys.getsizeof(self)))
            
        
        self.G = self.load_nx(self.nodes, self.edges, name='train')
        del(self.nodes)
        del(self.edges)
        
        n = gc.collect()        
        
    def chunkIt(self, seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out
   
    def walk_progress(self,source_node_types, target_node_types, initials, all_nw = 40, all_wl = 40, num_source = 0
                      , num_allow_loop = 10, walk_file_prefix = 'walks_nx'):
        
        num_thread = self.num_thread
        
        self.update_attrs(self.graph_pattern, self.type_to_node, self.loop_index)
        del(self.type_to_node)
        
        self.all_nw = all_nw
        self.all_wl = all_wl
        self.source_node_types = source_node_types
        self.target_node_types = target_node_types
        self.num_source = num_source
        start = time.time()
        
        self.initials = initials
        self.num_source = num_source
        self.num_allow_loop = num_allow_loop
        self.walk_file_prefix = walk_file_prefix
        
        num_source = 0
        if self.num_source != None:
            num_source = self.num_source
        
        graph_pattern = self.graph_pattern
        print(type(graph_pattern[0]))
        
        source_nodes = self.get_nodes_by_types(graph_pattern[0].keys())        
        target_nodes = self.get_nodes_by_types(graph_pattern[-1].keys())
        
        
        
        print('source nodes: {}, target_nodes: {}'.format(len(source_nodes), len(target_nodes)))
        

        selected_nodes = []
        if num_source == 0:
            selected_nodes = source_nodes
        else:
            selected_nodes = random.sample(source_nodes, num_source)

        print('source nodes: {}, selected_nodes: {}'.format(len(source_nodes), len(selected_nodes)))
        
        
        
        walk_file_prefix = 'walks_nx'
        if self.walk_file_prefix!=None:
             walk_file_prefix = walk_file_prefix

        walk_file = '{}_{}_{}_{}_{}_bwV4.csv'.format(walk_file_prefix, self.all_nw, self.all_wl, num_source, num_allow_loop)
        if self.postfix != '':
            walk_file = '{}_{}_{}_{}_{}_bwV4_{}.csv'.format(walk_file_prefix, self.all_nw, self.all_wl, num_source, num_allow_loop, self.postfix)
            
        job_number = cpu_count()
        
        source_node_lists = self.chunkIt(selected_nodes, job_number)
        print('job_number: {}, chunks for thread: {}'.format(job_number, len(source_node_lists)))
              
        jobs = []

        for i, s in enumerate(source_node_lists):
            j = multiprocessing.Process(target=self.walk, args=(walk_file, s))
            jobs.append(j)
        for j in jobs:
            j.start()
      
        
#         self.global_lock = threading.Lock()
        
#         threads = []
        
#         for i in range(num_thread):
#             t = threading.Thread(target=self.walk, args=(walk_file, source_node_lists[i]))
#             threads.append(t)
#             t.start()
#         [thread.join() for thread in threads]
        
#         walk_file = self.walk(self.graph_pattern, self.loop_index, self.type_loop, initials, self.all_nw, 
#                                     self.all_wl, num_source = num_source, num_allow_loop=num_allow_loop, walk_file_prefix=walk_file_prefix)

        end = time.time()
        print('time elapse: {}'.format(end - start))
        return walk_file
        
        
    def load_data(self):
        cols = ['h_id', 't_id', 'r_id', 'h_c_id', 't_c_id']
        
        df_train = pd.read_csv(self.data_path + self.train_file,usecols=cols)
        print('df_train: {}'.format(df_train.shape))
        
        ## check which nodes are included
        gp = self.graph_pattern
        sel_types = []
        for p in gp:
            sel_types.extend(list(p.keys()))
        set_sel_types = set(sel_types)
        
        
        df_train = df_train[(df_train['h_c_id'].isin(set_sel_types)) & (df_train['t_c_id'].isin(set_sel_types))]
        print('based on graphpattern df_train_sel: {}, set_sel_types: {}'.format(df_train.shape, len(set_sel_types)))

        h_nodes = set(tuple(zip(df_train.h_id, df_train.h_c_id)))
        t_nodes = set(tuple(zip(df_train.t_id, df_train.t_c_id)))
        print('h: {}, t: {}'.format(len(h_nodes), len(t_nodes)))
        self.nodes = list(set( list(h_nodes) + list(t_nodes) ))
        print('nodes: {}'.format(len(self.nodes)))
        
        del(h_nodes)
        del(t_nodes)
        
        self.edges = set(tuple(zip(df_train.h_id,df_train.t_id)))
        
        print('types of node and their counts:')
        self.type_to_node = dict()
        for i in self.nodes:
            if i[1] not in self.type_to_node:
                self.type_to_node[i[1]] = list()
            self.type_to_node[i[1]].append(i[0])
        print('type_to_node: {}'.format(len(self.type_to_node)))
        
        self.id_to_type = dict()
        for t in self.type_to_node:
            for n in self.type_to_node[t]:
                self.id_to_type[n] = t
        print('id_to_type: {}'.format(len(self.id_to_type)))   
        
        gc.collect()
        

    def load_nx(self, nodes, edges, name='train'):
        G = nx.Graph(name=name)
        
        for n in tqdm(nodes, desc = 'Nodes'):
            G.add_node(n[0], type=n[1], visited=0)
        for e in tqdm(edges, desc = 'Edges'):
            G.add_edge(*e)

#         print(nx.info(G))
        
        return G
    
    def find_indices(self, node ,graph_pattern, index, loop_allowed):
    
        node_type = self.id_to_type[node]
    

        gp_dict = dict()
        gp_len = len(graph_pattern)    
        for i in range(gp_len):
            keys = list(graph_pattern[i].keys())
            for k in keys:
                if k not in gp_dict:
                    gp_dict[k] = list()
                gp_dict[k].append(i)

        allowed_positions = gp_dict[node_type]
    
        gp = list(graph_pattern[index].keys())

        if index not in allowed_positions:
            print('SHOULD NOT HAPPEN!!!!!!!')
            return None

        # from now on we work with index
        next_index = (index + 1) % gp_len
        gp_next = list(graph_pattern[next_index].keys())

        nei_types = {self.id_to_type[nei] for nei in list(self.G[node]) }    

        if loop_allowed == False:
            if len([ i for i in nei_types if i in gp_next]) > 0 :
                return True
        else:
            if len([ i for i in nei_types if i in gp]) > 0 :
                return True 


        return False

    def update_attrs(self, graph_pattern, type_to_node, loop_index):
        override_count = 0

        allowed_dict = dict()
        skipping_nodes = set()
        skipping_nei = set()

        for i in tqdm(range(len(graph_pattern)-1, -1, -1),  desc = 'prepare allow dict'):
    
            cur_types = list(graph_pattern[i].keys())

            for type_ in cur_types:
    
                cur_list_nodes = list(type_to_node[type_])

                for node in cur_list_nodes:
                    loop_allowed = False
                    if i in loop_index:
                        loop_allowed = True

                    if node not in allowed_dict:
                        allowed_dict[node] = list()

                    if self.find_indices(node,graph_pattern, i, loop_allowed) == True:                    
                        allowed_dict[node].append(i)
        self.allowed_dict = allowed_dict
        
        ns = set()
        for i in allowed_dict:
            ns.add(self.id_to_type[i])
        print(ns)
        
        removed_counts = 0               
        for node in tqdm(self.G.nodes(), desc = 'check backwards'):
            if node not in allowed_dict:
                skipping_nodes.add(node)
                continue
                
            allowed_positions = allowed_dict[node]
            nei = list(self.G[node])
            nei_allowed = []
            for n in nei:
                try:
                    nei_allowed.extend(allowed_dict[n])
                except KeyError as e:
                    skipping_nei.add(n)

            remove_inx = list()
            for ap in allowed_positions:
                check_index = -1
                if ap in loop_index:
                    check_index = ap
                else:
                    check_index = (ap + 1) % len(graph_pattern)
    #                 print('check_index', check_index)
                    if check_index not in nei_allowed:
                        remove_inx.append(ap)


            for ri in remove_inx:
                allowed_dict[node].remove(ri)
                removed_counts+=1
                
        ns = set()
        for i in allowed_dict:
            
            ns.add(self.id_to_type[i])
        print(ns)

        # node is the destination 
        print('skipping_nodes: {}, skipping_nei: {}'.format(len(skipping_nodes), len(skipping_nei)))

        len_gp = len(graph_pattern)

        scores_dict = dict()
        for node in tqdm(self.G.nodes(),  desc = 'calc scores'):
            if node not in allowed_dict:
                
                continue

            score = dict()
            node_allowed = allowed_dict[node]
            node_type = self.G.nodes[node]['type']
            node_deg  = self.G.degree[node]  

            for index in range(len_gp):
                if index not in node_allowed:
                    score[index] = 0
                else:
                    gp = graph_pattern[index]
                    if self.degree_biased :
                        score[index] =1*  gp[node_type]/(node_deg)
                    else:
                        score[index] = gp[node_type]
                    

            scores_dict[node] = score

        node_nei_scores = dict()
        for node in tqdm(self.G.nodes(),  desc = 'assign scores of neighbours'):
            if node not in allowed_dict:
                
                continue

            nei = self.G[node]
            nei_scores = {}
            for index in range(len_gp):
                nei_index = [[n, scores_dict[n][index]] for n in nei]
                nei_scores[index] = np.array(nei_index)
            node_nei_scores[node] = nei_scores


        attrs = dict()         
        for n in allowed_dict:
            attrs[n] = {'allowed':  allowed_dict[n]}
            
        print('allowed attr prepared.')

        scores = dict()         
        for n in scores_dict:
            scores[n] = {'scores':  scores_dict[n]}
            
        print('scores attr prepared.')

        neighbours_scores = dict()
        for n in node_nei_scores:
            neighbours_scores[n] = {'nei_scores':  node_nei_scores[n]}
            
        print('nei_scores attr prepared')


        nx.set_node_attributes(self.G, attrs)
        nx.set_node_attributes(self.G, scores)
        nx.set_node_attributes(self.G, neighbours_scores)

        print('added attributes to the graph')
#         return override_count, attrs, removed_counts

    def get_nodes_by_types(self, types):
        return [x for x,y in self.G.nodes(data=True) if y['type'] in types]

    def select_next_GP_visited_biased(self, node, pnode, indx):
        all_d = self.G.nodes[node]['nei_scores'][indx] 
        
        nei = all_d[:, 0]
        p_index = np.where(all_d == pnode)
        nei = np.delete(nei, p_index)
#         p_index = nei.index(pnode) # newly added 
#         del(nei[p_index])# newly added 
        
        scores = all_d[:, 1]
        scores = np.delete(scores, p_index)
#         del(scores[p_index]) # newly added 
        if sum(scores) == 0:
            return None
        node = random.choices(nei, weights=scores, k=1)[0]
        return int(node)

    def walk(self, walk_file, selected_nodes):
#         while self.global_lock.locked():
#             sleep(0.01)
#             continue
        
#         self.global_lock.acquire()
        
        loop_count, exit_loop_count = 0, 0
        timeelapse = []
        graph_pattern = self.graph_pattern
        loop_index = self.loop_index
        type_loop = self.type_loop
        number_of_walks = self.all_nw
        walk_length = self.all_wl
        initials = self.initials
            
        num_allow_loop = 10
        if self.num_allow_loop != None:
            num_allow_loop = self.num_allow_loop
                          
        
        gp_indx = 1

        gp_len = len(graph_pattern)
     
        
            
        with open(self.data_path + walk_file, 'w') as outfile:

            for s in tqdm(selected_nodes):

                for nw in range(number_of_walks):
                    node = s
                    s_init_type = initials[self.G.nodes[node]['type']]


                    prev_node = ''
                    wl = 0
                    i  = 1

                    outline = ' ' + s_init_type + str(node)


                    while i <= gp_len:

        #                 start = time.time()

                        if i not in loop_index:   

                            if i == gp_len:
                                i = 0


                            next_node= self.select_next_GP_visited_biased(node, prev_node, i) # normal step

                            i +=1
                            wl+=1
                            if wl >= walk_length:

                                break

                            if next_node == None:

                                break
                            prev_node = node
                            node = next_node
                            outline += ' ' + initials[self.G.nodes[node]['type']] + str(node)

                        else:
                            loop_break = False
                            cp = 0

                            while True:

                                next_node = self.select_next_GP_visited_biased(node, prev_node, i)
                                cp+=1
                                wl+=1

                                if wl >= walk_length:
                                    loop_break = True

                                    break
                                if next_node == None:
                                    loop_break = True

                                    break
                                prev_node = node
                                node = next_node

                                outline += ' ' + initials[self.G.nodes[node]['type']] + str(node) 

                                loop_count +=1


                                if (cp >= num_allow_loop) and (self.exit_the_loop(node, graph_pattern,i+1, prev_node)):

                                    exit_loop_count+=1
                                    break                       

                            i+=1

                            if loop_break == True:
                                break

                    outfile.write(outline + '\n')

        print('loop_count: {}, exit_loop_count: {}'.format(loop_count, exit_loop_count))
#             outfile.close()
#         self.global_lock.release()

    def exit_the_loop(self, node, graph_pattern, indx, pnode):
        if indx >= len(graph_pattern):
            indx = 0
            
        gp = graph_pattern[indx]

        gp ={k for k,v in gp.items() if v>0}

        nei = [i for i in self.G[node]]

        nei_attr = [self.G.nodes[n] for n in nei]

        nei_gp = [nei[i] for i in range(len(nei)) if (nei_attr[i]['type'] in gp) and indx in nei_attr[i]['allowed']]

        if len(nei_gp) != 0:
    
            return True
        return False


