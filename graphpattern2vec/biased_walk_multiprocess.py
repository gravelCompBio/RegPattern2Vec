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

from multiprocessing import cpu_count, Queue
import multiprocessing

from collections import defaultdict


class BiasedWalk(object):

    data_path = 'data/'
    edge_file = 'df_edges.csv'
    train_file = 'df_train.csv'
    
    def __init__(self, graph_pattern, loop_index, type_loop,  data_path = 'data/',  train_file = 'df_train.csv', postfix = '', degree_biased = True, num_processes = 0):

        self.data_path = data_path
        
        self.train_file = train_file
        self.graph_pattern = graph_pattern
        self.loop_index = loop_index
        self.type_loop = type_loop
        self.postfix = postfix
        self.num_processes = num_processes
        
        print('loading data ...')
        self.load_data()
        
        self.degree_biased = degree_biased
                            
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
   
    def walk_progress(self, initials, all_nw = 40, all_wl = 40, num_source = 0
                      , num_allow_loop = 10, walk_file_prefix = 'walks_nx', loop_chance = .20):
        ## source_node_types, target_node_types,
        
        num_processes = self.num_processes
        
        self.update_attrs(self.graph_pattern, self.type_to_node, self.loop_index)
        del(self.type_to_node)
        
        self.all_nw = all_nw
        self.all_wl = all_wl

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
        
        
        source_nodes = self.get_nodes_by_types(graph_pattern[0].keys())        
        target_nodes = self.get_nodes_by_types(graph_pattern[-1].keys())
        
        
        
        print('source nodes: {}, target_nodes: {}'.format(len(source_nodes), len(target_nodes)))
        

        selected_nodes = []
        if num_source == 0:
            selected_nodes = source_nodes
        else:
            selected_nodes = random.sample(source_nodes, num_source)

        print('source nodes: {}, selected_nodes: {}'.format(len(source_nodes), len(selected_nodes)))
        
        
        
#         walk_file_prefix = 'walks_nx'
        if self.walk_file_prefix!=None:
             walk_file_prefix = walk_file_prefix

        walk_file = '{}_{}_{}_{}_{}_bwV4.csv'.format(walk_file_prefix, self.all_nw, self.all_wl, num_source, num_allow_loop)
        if self.postfix != '':
            walk_file = '{}_{}_{}_{}_{}_bwV4_{}.csv'.format(walk_file_prefix, self.all_nw, self.all_wl, num_source, num_allow_loop, self.postfix)
        
        if self.num_processes:
            job_number = self.num_processes
        else:     
            job_number = cpu_count()
        
                
        source_node_lists = self.chunkIt(selected_nodes, job_number)
        print('job_number: {}, chunks for thread: {}'.format(job_number, len(source_node_lists)))
              
        jobs = []
        q = Queue()
        
        for i, s in enumerate(source_node_lists):
            j = multiprocessing.Process(target=self.walk, args=(walk_file, s, q, loop_chance))
            jobs.append(j)
        for j in jobs:
            j.start()
      
        with open(self.data_path + walk_file, 'w') as outfile:
            for i in range(len(jobs)):
                results =  q.get()
                for result in results:
                    outfile.write(result + '\n')

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
        
        self.edges = set(tuple(zip(df_train.h_id,df_train.t_id, df_train.r_id)))
        
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
        
        
#         ## relation to count for biasing toward relation types with lower number
#         self.d_relation_to_count = defaultdict(int)
        
#         for i in df_train.itertuples():
#             _,_,_,r_id,_,_ = i
#             self.d_relation_to_count[r_id] += 1
          
#         print('self.d_relation_to_count: {}'.format(len(self.d_relation_to_count)))
        
        
        gc.collect()
        
#     def get_relation_probs(self, list_of_rels):
#         relation_scores = [1/self.d_relation_to_count[i] for i in list_of_rels]
        

#         sum_scores = sum(relation_scores)
#         relation_scores_normalized = []
#         total = 0
#         relation_scores_normalized = [score/sum_scores for score in relation_scores]
#         return relation_scores_normalized
        
        

    def load_nx(self, nodes, edges, name='train'):
        G = nx.Graph(name=name)
        
        for n in tqdm(nodes, desc = 'Nodes'):
            G.add_node(n[0], type=n[1], visited=0)
        for e in tqdm(edges, desc = 'Edges'):
            h,t,r = e
            G.add_edge(h,t, relation = r)

        
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
#         print(ns)
        
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
#         print(ns)

        # node is the destination 
        assert(len(skipping_nodes) == len(skipping_nei) == 0 )
        
#         print('skipping_nodes: {}, skipping_nei: {}'.format(len(skipping_nodes), len(skipping_nei)))

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
        #print(type(all_d))
        #print(type(pnode))
        #print()
        
        nei = np.delete(nei, p_index)                              
        
        scores = all_d[:, 1]
        scores = np.delete(scores, p_index)
        
        rels = [self.G[node][n]['relation'] for n in nei]       
        if len(rels) == 0:
            return None
        unique_rels = list(set(rels))
        
        rel_choosed = random.choice(unique_rels)
        selected_indices = [i for i in range(len(rels)) if rels[i] == rel_choosed]
        
        sel_nei_score = [ scores[i] for i in selected_indices]
        sel_nei = [ nei[i] for i in selected_indices]
                
        if sum(sel_nei_score) == 0:
            return None
        
        assert(len(sel_nei_score) == len(sel_nei))
        
        node = random.choices(sel_nei, weights=sel_nei_score, k=1)[0]
        return int(node)

    def walk(self, walk_file, selected_nodes,q, loop_chance):
        
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
     
        
        results = []
        

        for s in tqdm(selected_nodes):

            for nw in range(number_of_walks):
                node = s
                s_init_type = initials[self.G.nodes[node]['type']]


                prev_node = ''
                wl = 0
                i  = 1

                outline = ' ' + s_init_type + str(node)
                
                forw = True

                while i <= gp_len:

                    if i in loop_index and random.random() < loop_chance:   
#                         loop_break = False
                        cp = 0

                        while True:

                            next_node = self.select_next_GP_visited_biased(node, prev_node, i)
                            cp+=1
                            wl+=1

                            if wl >= walk_length:
#                                 loop_break = True
                                break
                                
                            if next_node == None:
#                                 loop_break = True
                                break
                                
                            prev_node = node
                            node = next_node

                            outline += ' ' + initials[self.G.nodes[node]['type']] + str(node) 

                            loop_count +=1


                            if (cp >= num_allow_loop) or (self.exit_the_loop(node, graph_pattern,i+1, prev_node)):
#                                 loop_break = True
                                break                                               

#                         if loop_break == True:
#                             break

                    else:  
                        
#                         if i == gp_len:
#                             i = 0
                                                       

                        next_node= self.select_next_GP_visited_biased(node, prev_node, i) # normal step

                        
                        wl+=1
                        if wl >= walk_length:
                            break

                        if next_node == None:
                            break
                            
                        prev_node = node
                        node = next_node
                        outline += ' ' + initials[self.G.nodes[node]['type']] + str(node)
                        
                        if forw and i < gp_len - 1:
                            i += 1
                        elif forw and i == gp_len - 1:
                            forw = False
                            i -= 1
                        elif not forw and i == 0:
                            forw = True
                            i += 1
                        elif not forw and i > 0:
                            i -= 1
                    
                    # i?
                    
                        
                    
                results.append(outline) #+ '\n')
        q.put(results)
        return results


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


