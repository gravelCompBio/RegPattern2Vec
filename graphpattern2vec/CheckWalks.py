import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import numpy as np
import re
import mmap
from collections import defaultdict


class CheckWalks():
    
    def __init__(self, data_path, train_file, node_file):
        
        """ mode : either node_type or relation_type """
        
        self.data_path = data_path               
        self.train_file = train_file    
        self.node_file = node_file
        
        print('load datas....')
        self.load_train_file()     
        self.load_node_file()
        self.load_edge_tuples()
                    
            
    def process(self, walk_file_name, mode = 'relation_type', edge_verification_for_walks = False, walk_length_plot = False):
        self.walk_file_name = walk_file_name
        self.mode = mode 
        self.edge_verification = edge_verification_for_walks
        
        
        if mode == 'relation_type':
            self.load_tup_to_relation()
            self.find_metapaths_relation()                
            
        elif mode == 'node_type':
            self.load_node_to_type()
            self.find_metapaths_nodeTypes()
            
        if self.edge_verification:
            self.check_walk_in_edge()
            
        if walk_length_plot == True:
            self.show_bar(list(self.d_wl_sorted.keys()), list(self.d_wl_sorted.values()))
            
        if mode == 'relation_type':              
            return self.metapaths_sorted
        elif mode == 'node_type':            
            return self.metapaths_name_sorted
        
    def find_metapaths_nodeTypes(self):

        d_mps_count = defaultdict(int)
        if self.edge_verification:
            edges = set()
        
        file_path = self.data_path + self.walk_file_name
        d_wl = defaultdict(int)
        d_mps_count = defaultdict(int)

        with open(file_path) as f:       
            for line in tqdm(f, total=self.get_num_lines(file_path)):

                lst = line.strip().split()
                d_wl[len(lst)] += 1

                ids = [int(i[1:]) for i in lst]
                initials = [i[0] for i in lst]
                s_initials = ''.join(initials)
                lst = [int(i[1:]) for i in lst]
                
                if self.edge_verification:
                    edges |= {(lst[j], lst[j+1]) for j in range(len(ids) - 1)}     

                

                index_list = self.find_indices(s_initials)

                for start, end, move in index_list:

                    sel = ids[int(start): int(end)]
                    if move == 'B':
                        sel.reverse()

                    mp = []
                    for i in range(len(sel)):
                        mp.append(self.node_to_type[sel[i]])

                    d_mps_count[tuple(mp)] += 1


        self.d_wl_sorted = dict(sorted(d_wl.items(), key=lambda item: item[0], reverse=True))  
        self.metapaths_sorted = dict(sorted(d_mps_count.items(), key=lambda item: item[1], reverse=True))  
        
        if self.edge_verification:
            self.walk_edges = edges
            print('self.walk_edges: {}'.format(len(self.walk_edges)))
            
        self.metapaths_name_sorted = dict()
        for mp in self.metapaths_sorted:
            lst = []
            for i in mp:                
                lst.append(self.nodeTypeId_to_name[i])
            self.metapaths_name_sorted[tuple(lst)] = self.metapaths_sorted[mp]                                     
        
    def find_metapaths_relation(self):
#         mps = []
        if self.edge_verification:
            edges = set()
        
        d_mps_count = defailtdict(int)
        d_wl = defaultdict(int)
        
        file_path = self.data_path + self.walk_file_name
        with open(file_path) as f:
            for line in tqdm(f, total=self.get_num_lines(file_path)):

                lst = line.strip().split(' ')
                d_wl[len(lst)] += 1
                initials = [i[0] for i in lst]
                s_initials = ''.join(initials)
                lst = [int(i[1:]) for i in lst]

                if self.edge_verification:
                    for j in range(len(lst) - 1):
                        edges.add((lst[j], lst[j+1]))

                index_list = self.find_indices(s_initials)
                if len(index_list) == 0:
#                     line = f.readline()
                    continue

                for start, end, move in index_list:

                    sel = lst[int(start): int(end)]
                    if move == 'B':
                        sel.reverse()

                    mp = []
                    for i in range(len(sel)-1):
                        tup = (sel[i], sel[i+1])
                        r, direction = self.tuple_to_relation[tup]
                        rel_name = r        
                        mp.append(rel_name)       

                    d_mps_count[tuple(mp)] += 1
        print('d_mps_count: {}'.format(len(d_mps_count)))
        self.d_wl_sorted = dict(sorted(d_wl.items(), key=lambda item: item[0], reverse=True))  
   
        
        self.metapaths_sorted = dict(sorted(d_mps_count.items(), key=lambda item: item[1], reverse=True))   
        print('self.metapaths_sorted: {}'.format(len(self.metapaths_sorted)))
        
        if self.edge_verification:
            self.walk_edges = edges
            print('self.walk_edges: {}'.format(len(self.walk_edges)))
        
    
    def get_num_lines(self, file_path):
        with open(file_path) as f:
            return sum(1 for _ in f)


    def load_train_file(self):
        self.df_train = pd.read_csv(self.data_path + self.train_file, dtype= {'t':'str'})
        print('self.df_train: {}'.format(self.df_train.shape))
        
    def load_edge_tuples(self):
        self.edge_tuples = set(zip(self.df_train.h_id, self.df_train.t_id ))
        print('self.edge_tuples: {}'.format(len(self.edge_tuples)))        
        
    def load_node_file(self):
        self.df_node = pd.read_csv(self.data_path + self.node_file)
        print('self.df_node: {}'.format(self.df_node.shape))
        
    def load_node_to_type(self):
        self.node_to_type = dict(zip(self.df_node.id, self.df_node.type_id))
        print('self.node_to_type: {}'.format(len(self.node_to_type)))
        self.nodeTypeId_to_name = dict(zip(self.df_node.type_id, self.df_node.type))
        print('self.nodeTypeId_to_name: {}'.format(len(self.nodeTypeId_to_name)))
        

    def load_tup_to_relation(self):
        tuple_to_relation = {}
        for i in tqdm(self.df_train.itertuples(), desc = 'creating edge tuples to relation dictionary'):
            _, _, _, r, h_id, t_id,_,_,_,_,_ = i

            tup = (h_id, t_id)
            rev_tup = (t_id, h_id)

            tuple_to_relation[tup] = (r, 'N')
            tuple_to_relation[rev_tup] = (r, 'R')
        
        self.tuple_to_relation = tuple_to_relation

        print('tuple_to_relation: {}'.format(len(self.tuple_to_relation)))
        
    def show_bar(self, x, y):
        _ = plt.bar(x, y)
              
    def check_walk_in_edge(self):
        
        lst = self.walk_edges # edge walks
        e_tup = self.edge_tuples # train edges
        
        c, t = 0, 0
        for i in tqdm(lst):

            head = i[0]
            tail = i[1]


            if ((head, tail) not in e_tup) and ((tail, head) not in e_tup):
                c+=1
    
            elif ((head, tail)  in e_tup) or ((tail, head)  in e_tup):
                t+=1

        print('not exist {} out of {}={}'.format(c, len(lst), t))
        
    def find_indices(self,s):
        forward = '^v[v|f|(vf)]+va'
        backward = '^av[v|f|(vf)]+v'
        i = 0
        prev = 0

        index_list = []

        while len(s):
            back = re.search(backward, s)
            forw = re.search(forward, s)

            if back:
                span = back.span()
                index_list.append((prev + span[0], prev + span[1], 'B'))
                prev += span[1]-1
                s = s[span[1]-1:]

            elif forw:
                span = forw.span()
                index_list.append((prev + span[0], prev + span[1], 'F'))
                prev += span[1]-1
                s = s[span[1]-1:]


            else:
                return index_list        

        return index_list     
    
#     def find_indices(self, s):
#         forward = '^v[v|f|(vf)]+va'
#         backward = '^av[v|f|(vf)]+v'
#         i = 0
#         prev = 0

#         index_list = []

#         while len(s):
#             back = re.search(backward, s)
#             forw = re.search(forward, s)

#             if back:
#                 span = back.span()
#                 index_list.append((prev + span[0], prev + span[1], 'B'))
#                 prev += span[1]-1
#                 s = s[span[1]-1:]

#             elif forw:
#                 span = forw.span()
#                 index_list.append((prev + span[0], prev + span[1], 'F'))
#                 prev += span[1]-1
#                 s = s[span[1]-1:]


#             else:
#                 return index_list        

#         return index_list
