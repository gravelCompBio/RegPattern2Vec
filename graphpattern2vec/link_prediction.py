import pandas as pd
from tqdm import tqdm
import numpy as np
import csv
import json
import random
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold


import statsmodels.api as sm
import math

import time
import pickle

import gc

# plt.style.use('ggplot')

# %matplotlib notebook

class LinkPrediction(object):
    
    def __init__(self, data_path, df_merge_file, df_nodes_file, df_relations_file, df_type_file, emb_file
                 , test_set, result_path, the_link, mp2v_emb = True):
        self.__data_path         = data_path
        self.__df_relations_file = df_relations_file
        self.__df_nodes_file     = df_nodes_file
        self.__df_merge_file     = df_merge_file 
        self.__df_types_file      = df_type_file
        self.__emb_file          = emb_file
        self.__result_path       = result_path
        self.__test_set          = test_set
        self.the_link            = the_link
        
        
        self.__merge_cols        = ['h','t','r','h_id','t_id', 'r_id', 'h_c', 't_c', 'h_c_id', 't_c_id']
        self.__merge_cols_names  = self.__merge_cols[:3]
        self.__merge_cols_ids    = self.__merge_cols[3:]
        self.__node_cols        = ['id', 'name', 'type']
        self.__relation_cols        = ['id', 'name']
        
        self.__load_files()
        self.__read_embeddings(mp2v_emb = mp2v_emb)
        self.__get_training_examples()
        
    def draw_pca_of_all_nodes(self):
        self.__label_all_node_embedding()
        self.__plot_pca (self.__labeled_emb, 'PCA-{}.tiff'.format(self.__emb_file))
        
    def __plot_pca(self, lst_emb, plot_name):
        
        df_emb2 = pd.DataFrame(lst_emb)
#         print(df_emb2.head(2))
        df_emb2_copy = df_emb2.copy()
        df_emb2.drop(df_emb2.columns[[-1]], axis=1, inplace=True)
        df_emb2 = df_emb2.dropna()
#         print(df_emb2.head(2))
        emb = df_emb2[:].values
        #target
        df_emb2_copy = df_emb2_copy.dropna()
        df = df_emb2_copy.iloc[:,-1]
        
        #print(emb[0][-1])


        pca = PCA(n_components = 2)
        principalComponents = pca.fit_transform(emb)
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = ['principal component 1', 'principal component 2'])

        
        finalDf = pd.concat([principalDf, df], axis = 1)
        finalDf.columns = ['principal component 1', 'principal component 2', 'target']
        # finalDf.head(2)

        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1', fontsize = 10)
        ax.set_ylabel('Principal Component 2', fontsize = 10)
        ax.set_title('2 component PCA', fontsize = 20)
        targets = [0,1,2]
        targets_name = ['Proteins', 'Pathway', 'others']
        colors = ['r', 'g','b']
        for target, color in zip(targets,colors):
            indicesToKeep = finalDf['target'] == int(target)
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c = color
                       , s = 50)

        ax.legend(targets_name)
        ax.grid()
        plt.savefig(self.__data_path + self.__result_path + plot_name, dpi=300)
        
    def train_model_cross_validation_with_split(self, solver, multi_class, penalty='l2', C=1e5, cv = 10, max_iter = 100, sel_feat = False, ver = ''):
        ''' we used it when training on whole graph'''
        
        lst_X = [i[:-1] for i in self.training_examples]
            
        lst_Y = [i[-1:] for i in self.training_examples]
        
        print('train_model_cross_validation| train_model| {} and {}'.format(len(lst_X[0]), len(lst_Y[0])))
        X_train, X_test, y_train, y_test = train_test_split(lst_X, lst_Y, test_size=0.3, random_state=0)
        
        print('train_model_cross_validation| train_model| X_train: {}, X_test: {}, y_train: {}, y_test:{}'.format(len(X_train), len(X_test), len(y_train), len(y_test)))
        
        y_all = np.asarray(lst_Y)
        y_train = np.asarray(y_train)
        
        
        self.logreg = LogisticRegression(C=C, solver=solver, penalty=penalty, multi_class=multi_class, max_iter = max_iter)
#         y_test_predict = self.logreg.fit(X_train, np.ravel(y_train)).decision_function(X_test)
        
        scores = cross_val_score(self.logreg, X_train, np.ravel(y_train), cv=cv)
        print('train_model_cross_validation| Cross-validation scores: {}'.format(scores ))
        print("train_model_cross_validation| Cross-validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
        self.logreg.fit(X_train, np.ravel(y_train))

        
        # to test
#         pred_test = self.logreg.predict(X_test)
        pred_test_prob = self.logreg.predict_proba(X_test)[:, 1]
    
        self.cuttoff = self.Find_Optimal_Cutoff(y_test, pred_test_prob)
        print('Cutt off is calculated to be {}'.format(self.cuttoff))
            
        pred_test = [1 if p>self.cuttoff else 0 for p in pred_test_prob ]

        confusionmatrix = confusion_matrix(y_test, pred_test)
        print( confusionmatrix)
        
        f1_score = sklearn.metrics.f1_score(y_test, pred_test)
        precision = sklearn.metrics.precision_score(y_test, pred_test)
        recall = sklearn.metrics.recall_score(y_test, pred_test)
        print('The test set, precision: {}, recall: {}, f1 score: {}'.format(precision, recall, f1_score))
        
        
        y_test = np.array(y_test)

        self.__AUC_ROC(np.ravel(y_test),pred_test_prob, ver)
        
        modelFileName = 'model/model_{}_{}_{}_{}'.format(solver, multi_class,cv, ver)
        pickle.dump(self.logreg, open(modelFileName, 'wb'))
        
    def RF_model_cross_validation(self, n_estimators = 10, criterion = 'gini', max_depth = None, min_samples_split = 2, min_samples_leaf=1, min_weight_fraction_leaf = 0.0, max_features = 'auto', max_leaf_nodes = None, min_impurity_decrease = 0.0, min_impurity_split = None, bootstrap = True, random_state = None, verbose=0, cv = 10, sel_feat = False, ver = ''):
        
        lst_X = [i[:-1] for i in self.training_examples]
            
        lst_Y = [i[-1:] for i in self.training_examples]
        
        print('RF_model_cross_validation| train_model| {} and {}'.format(len(lst_X[0]), len(lst_Y[0])))
        X_train, X_test, y_train, y_test = train_test_split(lst_X, lst_Y, test_size=0.3, random_state=0)
        
        print('RF_model_cross_validation| train_model| X_train: {}, X_test: {}, y_train: {}, y_test:{}'.format(len(X_train), len(X_test), len(y_train), len(y_test)))
        
        y_all = np.asarray(lst_Y)
        y_train = np.asarray(y_train)
        
        self.RFClassifier = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, min_weight_fraction_leaf = min_weight_fraction_leaf, max_features = max_features, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, min_impurity_split= min_impurity_split, bootstrap = bootstrap, random_state = random_state, verbose = verbose)
        

        
        scores = cross_val_score(self.RFClassifier, lst_X, np.ravel(y_all), cv=cv)
        print('RF_model_cross_validation| Test set scores: {}'.format(scores ))
        print("RF_model_cross_validation| Test set Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
        self.RFClassifier.fit(lst_X, np.ravel(y_all))
        modelFileName = 'model/RF_model_{}_{}_{}_{}_{}'.format(n_estimators, criterion, max_depth,cv, ver)
        
        pickle.dump(self.RFClassifier, open(modelFileName + '.pkl', 'wb'))

        lst_pred_prob = self.RFClassifier.predict_proba(lst_X)[:, 1]
        
        tup = list(zip(np.ravel(y_all), lst_pred_prob))
        print('y_all: {}, lst_pred_prob: {}, '.format(len(y_all), len(lst_pred_prob) ))
        
        with open('for_tjur_rs.txt', 'w') as fp:
            fp.write('\n'.join('%s %s' % x for x in tup))
        
        print('for_tjur_rs is saved!')
        
    def train_model_cross_validation(self, solver, multi_class, penalty = 'l2', C=1e5, cv = 10, sel_feat = False, ver = '', scoring=('accuracy', 'f1')):
        
        lst_X = np.array([i[:-1] for i in self.training_examples])
            
        lst_Y = np.array([i[-1:] for i in self.training_examples])
        
        
        
        self.logreg = LogisticRegression(C=C, solver=solver, penalty= penalty, multi_class=multi_class)

        print('c: {}, solver: {}, multi_class: {}'.format(C, solver, multi_class))
    
        scores = cross_validate(self.logreg, lst_X, np.ravel(lst_Y), cv=cv,
                        scoring=scoring,
                        return_train_score=True)
        print('')
        print('Cross Validation Scores:')
        for s in scoring:
            s_fn = 'test_' + s 
            print('train_model_cross_validation | {} : {}'.format(s_fn, scores[s_fn]) )
            print("train_model_cross_validation| Cross Validation %s score: %0.2f (+/- %0.2f)" % (s_fn, scores[s_fn].mean(), scores[s_fn].std() * 2))
            print('-----------------------')        
        
        self.logreg.fit(lst_X, np.ravel(lst_Y))
        modelFileName = 'model/model_{}_{}_{}_{}_{}'.format(solver, multi_class, C,cv, ver)
        
        pickle.dump(self.logreg, open(modelFileName + '.pkl', 'wb'))

        lst_pred_prob = self.logreg.predict_proba(lst_X)[:, 1]
        
        tup = list(zip(np.ravel(lst_Y), lst_pred_prob))
        print('y_all: {}, lst_pred_prob: {}, '.format(len(lst_Y), len(lst_pred_prob) ))
        
        with open('for_tjur_rs.txt', 'w') as fp:
            fp.write('\n'.join('%s %s' % x for x in tup))
        
        print('for_tjur_rs is saved!')
        df_tjur_rs = pd.DataFrame(tup, columns=['label', 'prob'])
        

        self.plot_probability_distribuitn_by_labels(df_tjur_rs, plot_name = 'prob_distr_by_label_' + ver)

    def Mean_Reciprocal_Rank(self):
        # https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832
        #
        print('MRR|')
        the_link = self.the_link
        h_link = the_link[0]
        t_link = the_link[1]
        
        list_yes_tuple = list()
        for h in h_link:
            for t in t_link:
                tup = (h,t)
                lst = self.d_tuple_to_edge[tup]
                list_yes_tuple.extend(lst)
        print(len(list_ye))
        

        
    def plot_probability_distribuitn_by_labels(self, df, colors = ['red', 'green'], map_labels_to_names = {0.0:'Negative', 1.0:'Positive'}, plot_name = 'prob_distr_by_label'):
        
        labels = list(df.label.unique())


        fig, ax = plt.subplots(dpi=100)
        
        plt.title('Predicted probability based on Labels')
        plt.xlabel('Predicted Probabilities')
        plt.ylabel('Frequency (log scale)')
        
        
        handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in colors]
        
        label_names = [map_labels_to_names[i] for i in labels]

        ax.legend(handles, label_names)
        x = [df.loc[df.label == x, 'prob'] for x in labels]

        _ = ax.hist(x,  histtype='bar', color=colors, label=labels, log=True )
        
   
        plt.savefig(self.__data_path + self.__result_path + plot_name, dpi=300)
        
#         p, r, th = self.__precision_recall_curve(y_test ,y_test_predict, ver)  
#         self.__AUC_ROC(y_test ,y_test_predict, ver)
        

#         self.logreg.fit(X_train, np.ravel(y_train))
#         y_test_predict = self.logreg.predict(X_test)
#         y_score_lr = self.logreg.fit(X_train, y_train).decision_function(X_test)
#         y_test_predict = self.logreg.predict(X_test)
        
        
        
        
    
#         print('train_model_cross_validation| Accuracy of logistic regression classifier on test set: {:.2f}'.format(self.logreg.score(X_test, y_test)))
#         confusionmatrix = confusion_matrix(y_test, y_test_predict)
#         print(confusionmatrix)
        
#         print(classification_report(y_test, y_test_predict))
    
    def gridSearchLR(self):
        
        lst_X = [i[:-1] for i in self.training_examples]
            
        lst_Y = [i[-1:] for i in self.training_examples]
        
#         print('train_model_cross_validation| train_model| {} and {}'.format(len(lst_X[0]), len(lst_Y[0])))
#         X_train, X_test, y_train, y_test = train_test_split(lst_X, lst_Y, test_size=0.3, random_state=0)
        
#         print('train_model_cross_validation| train_model| X_train: {}, X_test: {}, y_train: {}, y_test:{}'.format(len(X_train), len(X_test), len(y_train), len(y_test)))
        
#         y_all = np.asarray(lst_Y)
#         y_train = np.asarray(y_train)
        
        model = LogisticRegression()
        solvers = ['newton-cg', 'lbfgs', 'liblinear']
        penalty = ['l2']
        c_values = [100, 10, 1.0, 0.1, 0.01]
        
        # define grid search
        grid = dict(solver=solvers,penalty=penalty,C=c_values)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
        grid_result = grid_search.fit(lst_X, lst_Y)
        
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    
    def gridSearchCV(self, thing = True):
        
        lst_X = [i[:-1] for i in self.training_examples]
            
        lst_Y = [i[-1:] for i in self.training_examples]
        
        print('gridSearchCV| train_model| {} and {}'.format(len(lst_X[0]), len(lst_Y[0])))
        X_train, X_test, y_train, y_test = train_test_split(lst_X, lst_Y, test_size=0.3, random_state=0)
        
        print('gridSearchCV| train_model| X_train: {}, X_test: {}, y_train: {}, y_test:{}'.format(len(X_train), len(X_test), len(y_train), len(y_test)))
        
        y_all = np.asarray(lst_Y)
        y_train = np.asarray(y_train)
        
        tuned_parameters = [
            {'classifier' : [LogisticRegression()],
             'classifier__penalty' : ['l1', 'l2'],
             'classifier__C' : np.logspace(-4, 4, 8),
              'classifier__solver' : ['liblinear']},
            {'classifier' : [RandomForestClassifier()],
            'classifier__n_estimators' : list(range(10,61,10)),
            'classifier__max_features' : list(range(6,22,5))}
        ]
        
        
        pipe = Pipeline([('classifier' , LogisticRegression())])
        
#         clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)
        
        scores = ['precision', 'recall', 'f1_score']
        
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(
                pipe, tuned_parameters, scoring='%s_macro' % score
            )
            clf.fit(X_train,  np.ravel(y_train))

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()

            
            
#         self.logreg = LogisticRegression(C=C, solver=solver, multi_class=multi_class)
# #         y_test_predict = self.logreg.fit(X_train, np.ravel(y_train)).decision_function(X_test)
#         print('c: {}, solver: {}, multi_class: {}'.format(C, solver, multi_class))
#         scores = cross_val_score(self.logreg, lst_X, np.ravel(y_all), cv=cv)
#         print('train_model_cross_validation| Test set scores: {}'.format(scores ))
#         print("train_model_cross_validation| Test set Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
#         self.logreg.fit(lst_X, np.ravel(y_all))
#         modelFileName = 'model_{}_{}_{}_{}_{}'.format(solver, multi_class, C,cv, ver)
        
#         pickle.dump(self.logreg, open(modelFileName + '.pkl', 'wb'))

#         lst_pred_prob = self.logreg.predict_proba(lst_X)[:, 1]
        
#         tup = list(zip(np.ravel(y_all), lst_pred_prob))
#         print('y_all: {}, lst_pred_prob: {}, '.format(len(y_all), len(lst_pred_prob) ))
        
#         with open('for_tjur_rs.txt', 'w') as fp:
#             fp.write('\n'.join('%s %s' % x for x in tup))
        
#         print('for_tjur_rs is saved!')
        
    def Logit(self, th = 0.5):
        
        lst_X = np.array([i[:-1] for i in self.training_examples])
            
        lst_Y = np.array([i[-1:] for i in self.training_examples])
        
        print('train_model_cross_validation| train_model| {} and {}'.format(len(lst_X[0]), len(lst_Y[0])))
        X_train, X_test, y_train, y_test = train_test_split(lst_X, lst_Y, test_size=0.3, random_state=0)
        
        print('train_model_cross_validation| train_model| X_train: {}, X_test: {}, y_train: {}, y_test:{}'.format(len(X_train), len(X_test), len(y_train), len(y_test)))
        
        lst_Y = lst_Y.astype(float)

        self.logit = sm.Logit(lst_Y,lst_X)
        self.logit_result = self.logit.fit()
        print('Results summary:')
        print(self.logit_result.summary())
        
        self.y_predicted = self.logit_result.predict(lst_X)

        predictions_nominal = [ 0. if x < th else 1. for x in self.y_predicted]

        lst_Y = lst_Y.reshape(-1)

        print(confusion_matrix(lst_Y, predictions_nominal))

        f1_score = sklearn.metrics.f1_score(lst_Y, predictions_nominal)
        precision = sklearn.metrics.precision_score(lst_Y, predictions_nominal)
        recall = sklearn.metrics.recall_score(lst_Y, predictions_nominal)
        print('The test set, precision: {}, recall: {}, f1 score: {}'.format(precision, recall, f1_score))
        
    
    def logit_evaluate_test_set(self, ver= ''):
        self.__load_test_set(self.__test_set)
        
        self.__get_test_set_emb(self.__df_test_set)
        self.__prepare_test_set_emb()
        test =  self.__test(self.__lst_test_set_all, ver = ver)
    
    def evaluate_test_set(self, ver= '', model = 'Logit', manual_cuttoff = 0):
        self.__load_test_set(self.__test_set)
        
        self.__get_test_set_emb(self.__df_test_set)
        self.__prepare_test_set_emb()
        self.save_list_of_test_embs(self.__lst_test_set_all, self.__lst_test_set_ids)
        test =  self.__test(self.__lst_test_set_all, ver = ver, model = model, manual_cuttoff = manual_cuttoff)
        return test
        
    def save_list_of_test_embs(self, lst_emb, ids):
        
        if len(lst_emb) != len(ids):
            print('save_list_of_test_embs| the lenghts from list of embeddings and list of ids are not matched')
            return 
        
        lst = []
        for i in range(len(lst_emb)):
            if ids[i][2] == lst_emb[i][-1]:
                tmp = ids[i][:2]
                tmp.extend(lst_emb[i])
                lst.append(tmp)
            else:
                print('save_list_of_test_embs| the labels from list of embeddings and list of ids are not matched')
                return 
        print('save_list_of_test_embs| lst: {} to save.'.format(len(lst)))
        with open(self.__data_path + 'test_set_embeddings.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(lst)
        print('save_list_of_test_embs| {} is saved.'.format(self.__data_path + 'test_set_embeddings.csv'))
        
        
    def __prepare_test_set_emb(self):
        
        list_test_set_yes_emb = [ [i[1], i[2]] for i in self.__lst_test_set]
        self.__lst_test_set_all = list()
        for i in list_test_set_yes_emb:
            lst = list([])
            lst.extend(i[0])
            lst.append(i[1])
            self.__lst_test_set_all.append(lst)
        
        print('__prepare_test_set_emb| list of test set prepared {}'.format(len(self.__lst_test_set_all)))
        
    def __get_test_set_emb(self, df):
        err = 0
        self.__lst_test_set = list()
        self.__lst_test_set_ids = list()
        
        mcols = self.__merge_cols
        
        
        for i in df.itertuples():
            
            try:
                indx = i[0]
                h_id = i[4]
                t_id = i[5]
                h_emb = self.__emb_dict[h_id]
                t_emb = self.__emb_dict[t_id]
                emb = np.multiply(h_emb, t_emb)
                self.__lst_test_set.append([indx, emb, 1])
                self.__lst_test_set_ids.append([h_id,t_id, 1])
                    
            except KeyError as e :
                print(e)
                err+=1
        list_h_id = list(df[mcols[3]].unique())
        list_t_id = list(df[mcols[4]].unique())
        
        test_edges = { (i[4],i[5]):i[0] for i in df.itertuples()}
        n_count = 0
        
        n_err = 0
        set_chk = set()
        for i in range(df.shape[0] ):
            n_h_id = random.choice(list_h_id)
            n_t_id = random.choice(list_t_id)
            set_chk.add(((n_h_id, n_t_id) not in test_edges) and ((n_h_id, n_t_id) not in self.d_tuple_to_edge))
            if ((n_h_id, n_t_id) not in test_edges) and ((n_h_id, n_t_id) not in self.d_tuple_to_edge):
                try:
                    n_h_emb = self.__emb_dict[n_h_id]
                    n_t_emb = self.__emb_dict[n_t_id]
                    n_emb = np.multiply(n_h_emb, n_t_emb)

                    n_index = i + df.shape[0]
                    self.__lst_test_set.append([n_index, n_emb, 0])
                    self.__lst_test_set_ids.append([n_h_id,n_t_id, 0])
                    n_count+=1
                except KeyError as e:
                    continue
            else:
                n_err+=1
            
        print('__get_test_set_emb| test set emb list : positive {} with {} errors, negative: {}, with {} errors.'
              .format(len(self.__lst_test_set) - n_count,err, n_count, n_err))
        
        print(set_chk)
        
    def __load_test_set(self, test_file):
        mcols = self.__merge_cols
        dtype = {mcols[0]:object, mcols[1]:object, mcols[2]:object, mcols[3]:int, mcols[4]:int, mcols[5]:int}
        self.__df_test_set     = pd.read_csv(self.__data_path + test_file ,dtype=dtype)
        print('__load_test_set| df_test_set: {}'.format(self.__df_test_set.shape))
        h = self.__df_test_set['h'].unique()
        t = self.__df_test_set['t'].unique()
        print('__load_test_set| unique heads: {}, unique tails: {}'.format(len(h), len(t)))
        
    def get_prediction_for_all(self,path_org_data, gene_map_file, reactome_map_file, n_iter = 1, p = 0.5, alternative='two-sided', pvalue = 0.05):
        
        self.__path_org_data, self.__gene_map_file, self.__reactome_map_file = path_org_data, gene_map_file, reactome_map_file
        
        # generate id for each possible pairs
        self.generate_dict_possible_pairs()
        
        c=0
        # add prediction for each pair in dictionary
        dict_ppid_emb = self.__dict_ppid_emb
        for i in tqdm(range(n_iter)):
            for k in dict_ppid_emb:
                
                prediction = self.logreg.predict_proba([dict_ppid_emb[k]])
                self.__dict_ppid_pred[k].append(prediction)
        
        dict_pair_prob = dict()
        for i in self.__dict_ppid_pred:
            pair = self.__dict_id_pp[i]
            dict_pair_prob[pair] = self.__dict_ppid_pred[i]
        
        self.save_dict_to_csv(dict_pair_prob, self.__emb_file)
        
    def save_dict_to_csv(self, d, file_name):
        w = csv.writer(open(self.__data_path + self.__result_path + 'dict_pred_prob_{}'.format(file_name), "w"))
        for key, val in d.items():
            w.writerow([key, val])
    
        
    def predict_for_specific_types(self, source_type, target_type, filter_source = None, filter_targets = None):
        
        node_type_to_id = self.node_type_to_id
        
        source_nodes = node_type_to_id[source_type]
        target_nodes = node_type_to_id[target_type]
        
        if filter_source != None:
            source_nodes = [i for i in source_nodes if i in filter_source]
        
        if filter_targets != None:
            target_nodes = [i for i in target_nodes if i in filter_targets]
        
        print('predict_for_specific_types| source_nodes: {}, target_nodes: {}'.format(len(source_nodes), len(target_nodes)))
        
        result = []
        neg = []
        all_results_prob = []
        
        all_count = 0 
        predicted_false_count = 0
        predicted_true_count = 0
        
        cpro = set()
        cpath = set()
        
        for s in tqdm(source_nodes): 
            all_count+=1
            if s in self.__emb_dict:
                
                source_emb = self.__emb_dict[s]
                for t in target_nodes:
                    
                    if t in self.__emb_dict:
                        target_emb = self.__emb_dict[t]
                        
                        e = np.multiply(source_emb, target_emb)
                        

                        prob = self.logreg.predict_proba([e])

                        if prob[0][1] > self.cuttoff:
                            prediction=1
                        else:
                            prediction=0
                       
                        odds = prob[0][1]/(1-prob[0][1])
                        logOdds = math.log(odds)
                        all_results_prob.append([s, t, prob[0][1], odds, logOdds])
                        
                        if (prediction ==1):
                            odds = prob[0][1]/(1-prob[0][1])
                            logOdds = math.log(odds)
                            result.append([s, t, prob, odds, logOdds])
                            predicted_true_count += 1
                        else:
                            neg.append([s, t, prob])
                            predicted_false_count+=1
                    else:
                        cpath.add(t)

            else:
                cpro.add(s)
        print('predict_for_specific_types| used cuttoff: {}'.format(self.cuttoff))
        print('predict_for_specific_types| missing sources: {}, missing targets: {}'.format(len(cpro), (cpath)))
        print('predict_for_specific_types| possitive predictions: {} (negative predictions: {})'.format(len(result), predicted_false_count))
        
        
        result.sort(key=lambda elem: elem[2][0][1], reverse=True)
        predict_results_list = [(i[0], i[1],i[2][0][1], i[3], i[4]) for i in result]
        predict_results_list = sorted(predict_results_list, key=itemgetter(2), reverse=True)
        
        print('predict_for_specific_types| removing the known predictions...')
        tup_to_e = self.d_tuple_to_edge
        edges = tup_to_e[(source_type, target_type)]
        
        id_to_name = self.node_id_to_name
        
        list_prediction_for_unknown = [(id_to_name[i[0]], id_to_name[i[1]], str(i[2]), str(i[3]), str(i[4])) for i in predict_results_list if (i[0], i[1]) not in edges]

        remained = len(predict_results_list) - len(list_prediction_for_unknown)
        print('predict_for_specific_types| all prediction: {}, prediction for unkowns: {}, remained(known): {}'.format(len(predict_results_list), len(list_prediction_for_unknown), remained))
        
        self.__list_prediction_for_unknown = list_prediction_for_unknown
        self.__save_list_3cols(list_prediction_for_unknown, self.__emb_file)
        print('predict_for_specific_types| Raw Predictions saved.')
        print()
                                            
        
        df_all_results_prob = pd.DataFrame(all_results_prob, columns=[source_type, target_type, 'probability', 'odds', 'logOdds'])
        all_resutls_path_name = self.__data_path + self.__result_path + 'pred_all_prob_{}.csv'.format(self.__emb_file)
        df_all_results_prob.to_csv(all_resutls_path_name, index=False)
        print('predict_for_specific_types| all predictions {} saved.'.format(df_all_results_prob.shape))
                                                 
        return pd.DataFrame(list_prediction_for_unknown, columns=[source_type, target_type, 'probability', 'odds', 'logOdds']), predict_results_list, neg
    
    def __save_list_3cols(self, lst, file_name):
#         print(lst[0])
        with open(self.__data_path + self.__result_path + 'predictions_raw_{}.csv'.format(file_name), 'w') as fp:
            fp.write('\n'.join('%s,%s,%s,%s,%s' % x for x in lst))
    
    def __load_files(self):
        mcols = self.__merge_cols
        dtype = {mcols[0]:object, mcols[1]:object, mcols[2]:object, mcols[3]:int, mcols[4]:int, mcols[5]:int}
        self.__df_merge     = pd.read_csv(self.__data_path + self.__df_merge_file ,dtype=dtype)
        self.__df_nodes     = pd.read_csv(self.__data_path + self.__df_nodes_file)
        self.__df_types     = pd.read_csv(self.__data_path + self.__df_types_file)
        self.__df_relations = pd.read_csv(self.__data_path + self.__df_relations_file)
            
        print('__load_files| df_merge: {}, df_nodes: {}, df_relations: {}, df_types: {}'.format(self.__df_merge.shape, self.__df_nodes.shape
                                                                                  , self.__df_relations.shape, self.__df_types.shape))
        
        
#         self.type_name_to_id = dict(tuple(zip(self.__df_types.name, self.__df_types.id)))
        
        self.node_id_to_name = dict(tuple(zip(self.__df_nodes.id, self.__df_nodes.name))) 
#         self.node_name_to_id = dict(tuple(zip(self.__df_nodes.name, self.__df_nodes.id)))
        self.node_id_to_type = dict(tuple(zip(self.__df_nodes.id, self.__df_nodes.type)))
        
        node_type_to_id = dict()
        for i in self.node_id_to_type:
            type_ = self.node_id_to_type[i]
            if type_ not in node_type_to_id:
                node_type_to_id[type_] = list()
            node_type_to_id[type_].append(i)
        self.node_type_to_id = node_type_to_id
        
        del(self.node_id_to_type)
        
#         print('__load_files| type_name_to_id: {}, node_name_to_id: {}, node_id_to_type: {}, node_type_to_id: {}'.format(len(self.type_name_to_id), len(self.node_name_to_id),
#                                                                                     len(self.node_id_to_type), len(self.node_type_to_id)))
        
        self.relation_name_to_id = dict(tuple(zip(self.__df_relations.name, self.__df_relations.id)))
        self.relation_id_to_name = dict(tuple(zip(self.__df_relations.id, self.__df_relations.name)))
        
        # self.__merge_cols        = ['h','t','r','h_id','t_id', 'r_id', 'h_c', 't_c', 'h_c_id', 't_c_id']
        d_relation_to_edge = dict()
        d_tuple_to_edge = dict()
        for i in self.__df_merge.itertuples():
            h_id = i[4]
            t_id = i[5]
            r = i[3]
            h_c = i[7]
            t_c = i[8]
            
            tup = (h_c, t_c)
            
            if r not in d_relation_to_edge:
                d_relation_to_edge[r] = list()
            d_relation_to_edge[r].append((h_id, t_id))
            
            
            if tup not in d_tuple_to_edge:
                d_tuple_to_edge[tup] = list()
            d_tuple_to_edge[tup].append((h_id, t_id))
            
#         self.d_relation_to_edge = d_relation_to_edge
        self.d_tuple_to_edge    = d_tuple_to_edge
#         print('__load_files| d_relation_to_edge: {}, d_tuple_to_edge: {}'.format(len(self.d_relation_to_edge), len(self.d_tuple_to_edge)))        
        
        print()
    

        
    def __get_training_examples(self):
        print('Getting examples')
        the_link = self.the_link
        h_link = the_link[0]
        t_link = the_link[1]
        
        list_yes_tuple = list()
        for h in h_link:
            for t in t_link:
                tup = (h,t)
                lst = self.d_tuple_to_edge[tup]
                list_yes_tuple.extend(lst)
        yes_error = 0
        self.__list_yes_emb = list()
        for i in list_yes_tuple:
            try:
                arr = np.multiply(self.__emb_dict[i[0]], self.__emb_dict[i[1]])
                arr = np.append(arr,1)
                self.__list_yes_emb.append(arr )
            except KeyError as e:
                yes_error +=1 
                
        print('__get_training_examples| list_yes_tuple: {}, list_yes_emb: {} with {} errors.'.format(len(list_yes_tuple), len(self.__list_yes_emb), yes_error))
        
        node_type_to_id = self.node_type_to_id
        
        h_uniq_nodes = list()
        
     
        
        for h in h_link:
            h_uniq_nodes.extend(node_type_to_id[h])
        
        t_uniq_nodes = list()
        for t in t_link:
            t_uniq_nodes.extend(node_type_to_id[t])
            
        print('__get_training_examples| h_uniq_nodes: {}, t_uniq_nodes: {}'.format(len(h_uniq_nodes), len(t_uniq_nodes)))
        
        set_yes_edges = set(list_yes_tuple)
        rejected, no_err = 0,0
        self.__list_no_emb = list()
        
        for i in range(len(self.__list_yes_emb)):
            h = random.choice(h_uniq_nodes)
            t = random.choice(t_uniq_nodes)
            tup = (h, t)
            if tup not in set_yes_edges:
                try:
                    arr= np.multiply(self.__emb_dict[h], self.__emb_dict[t])
                    arr = np.append(arr,0)
                    self.__list_no_emb.append(arr )
                except KeyError as e:
                    no_err+=1
                    continue
            else:
                rejected += 1
        print('__get_training_examples| list_no_emb: {} with {} rejected, and {} errors.'.format(len(self.__list_no_emb), rejected, no_err))
        
        del(set_yes_edges)
        del(list_yes_tuple)
        print('__get_training_examples| release some data [set_yes_edges, list_yes_tuple]')
        
        self.__list_train_emb = self.__list_yes_emb
        self.__list_train_emb.extend(self.__list_no_emb)
        random.shuffle(self.__list_train_emb)
        print('__get_training_examples| shuffled.')
        
        del(self.__list_no_emb)
        del(self.__list_yes_emb)
        print('__get_training_examples| release some data[ __list_no_emb, __list_yes_emb]')
    
        gc.collect()
        print('__get_training_examples| Garbage Collector.')
        
#         lst_all = list()
#         for i in self.__list_train_emb:

#             lst = list([])
#             lst.extend(i[0])
#             lst.append(i[1])
            
#             lst_all.append(lst)
#         self.training_examples = lst_all

        self.training_examples = self.__list_train_emb
        
        del(self.__list_train_emb)
        print('__get_training_examples| release some data[__list_train_emb] ')
        
        print('__get_training_examples| Training data generated.')
        print()
        
        
    def __label_all_node_embedding(self):
        
        emb_all_nodes = self.__emb_dict

        lst_emb = []
        lst_name = []
        
        h_link = self.the_link[0]
        t_link = self.the_link[1]
        
        node_type_to_id = self.node_type_to_id
        
        h_uniq_nodes = list()
        for h in h_link:
            h_uniq_nodes.extend(node_type_to_id[h])
        
        t_uniq_nodes = list()
        for t in t_link:
            t_uniq_nodes.extend(node_type_to_id[t])
        
        h_uniq_nodes_set = set(h_uniq_nodes)
        t_uniq_nodes_set = set(t_uniq_nodes)
        print('__label_all_node_embedding| h_uniq_nodes_set: {}, t_uniq_nodes_set: {}'.format(len(h_uniq_nodes_set), len(t_uniq_nodes_set) ))

        for k in emb_all_nodes:  
            lst = []
            if (k != '/s>'):
                label = 2
                
                if int(k) in h_uniq_nodes_set:
                    label = 0
                elif int(k) in t_uniq_nodes_set:
                    label = 1

                lst = [i for i in emb_all_nodes[k]]
                lst.append(label)
                lst_name.append(k)
                lst_emb.append(lst)
        print('__label_all_node_embedding| shape label embeddings: {} and {}'.format(len(lst_emb), len(lst_emb[0]) ))

        self.__labeled_emb = lst_emb
                
    def __precision_recall_curve(self, y_test, y_scores_lr , ver=''):

        precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
        closest_zero = np.argmin(np.abs(thresholds))
        closest_zero_p = precision[closest_zero]
        closest_zero_r = recall[closest_zero]

        plt.figure(figsize=(6,6))
        plt.xlim([0.0, 1.01])
        plt.ylim([0.0, 1.01])
        plt.plot(precision, recall, label='Precision-Recall Curve')

        plt.xlabel('Precision', fontsize=16)
        plt.ylabel('Recall', fontsize=16)
        plt.axes().set_aspect('equal')
        plt.show()
        plt.savefig(self.__data_path + self.__result_path + 'precision_recall_curve' + ver)
        
        return precision, recall, thresholds
    
    def __hist_of_prob(self, probs):
        plt.hist(probs)
        plt.title('histogram of probabilities')
    
    def __AUC_ROC(self, y_test, y_score_lr, ver):
        
        self.__hist_of_prob(y_score_lr)
        
        
        fpr_lr, tpr_lr, ths = roc_curve(y_test, y_score_lr)
        
        d_for_roc_auc = {'fpr': list(fpr_lr), 'tpr': list(tpr_lr), 'threshold': list(ths)}
        

        with open(self.__data_path + 'auc_roc_' + ver[:-4] + '.json', 'w') as fp:
            json.dump(d_for_roc_auc, fp)
        
        self.thresholds = fpr_lr, tpr_lr, ths
        roc_auc_lr = auc(fpr_lr, tpr_lr)
        
        print('__AUC_ROC| roc_auc_lr: {}'.format(roc_auc_lr))

        #plt.figure(figsize=(8,7))
        plt.figure(figsize=(8,8))
        plt.xlim([-0.01, 1.00])
        plt.ylim([-0.01, 1.01])
        #plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
        plt.xlabel('False Positive Rate', fontsize=24)
        plt.ylabel('True Positive Rate', fontsize=24)
        plt.title('ROC curve', fontsize=24)
        
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        #
        plt.axes().set_aspect('equal')
        plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
        
        plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
        plt.legend(loc='lower right', fontsize=15)
        plt.savefig(self.__data_path + self.__result_path + 'roc_auc_lr' + ver)
        plt.show()
        
        return roc_auc_lr
        
    
    def __test(self, lst_to_test, ver = '_test_set', model = 'Logit', manual_cuttoff = 0):        
        
        lst_X = [i[:-1] for i in lst_to_test]
        lst_y = [i[-1:] for i in lst_to_test]
        
        print('__test| {} and {}'.format(len(lst_X), len(lst_y)))
        print('__test| {} and {}'.format(len(lst_X[0]), len(lst_y[0])))
        
        if model == 'Logit':

            lst_pred = self.logreg.predict(lst_X)
            lst_pred_prob = self.logreg.predict_proba(lst_X)[:, 1]
        elif model == 'RF':
            lst_pred = self.RFClassifier.predict(lst_X)
            lst_pred_prob = self.RFClassifier.predict_proba(lst_X)[:, 1]
            
        lst_y = np.array(lst_y)

        model_auc_roc = self.__AUC_ROC(np.ravel(lst_y),lst_pred_prob, ver)
        
        if manual_cuttoff == 0:
            self.cuttoff = self.Find_Optimal_Cutoff(lst_y, lst_pred_prob)[0]
            print('Cutt off is calculated to be {}'.format(self.cuttoff))
        else:
            self.cuttoff = manual_cuttoff
            print('Manual Cutt off set to be {}'.format(self.cuttoff))
        
        lst_pred_cuttoff = [ 1 if prob > self.cuttoff else 0 for prob in lst_pred_prob]
        
        
        confusionmatrix = confusion_matrix(lst_y, lst_pred_cuttoff)
        print( confusionmatrix)
        
#         print(lst_pred_prob[:10])
        
        f1_score = sklearn.metrics.f1_score(lst_y, lst_pred_cuttoff)
        precision = sklearn.metrics.precision_score(lst_y, lst_pred_cuttoff)
        recall = sklearn.metrics.recall_score(lst_y, lst_pred_cuttoff)
        print('The test set, precision: {}, recall: {}, f1 score: {}'.format(precision, recall, f1_score))
        
        print(classification_report(lst_y, lst_pred_cuttoff))
        
        
        
#         self.lst_X = lst_X
#         self.lst_y = lst_y
        
#         if model == 'logit':
#             lst_pred_prob = self.logreg.predict_proba(lst_X)[:, 1]
#         elif model == 'RF':
#             lst_pred_prob = self.RFClassifier.predict_proba(lst_X)[:, 1]
        
        tup = list(zip(np.ravel(lst_y), lst_pred_prob))
        print('y_all: {}, lst_pred_prob: {}, '.format(len(lst_y), len(lst_pred_prob) ))
        
        with open('for_tjur_rs_test.txt', 'w') as fp:
            fp.write('\n'.join('%s %s' % x for x in tup))
        
        print('for_tjur_rs_test is saved!')
        
        
   
        
       
        return precision, recall, f1_score, model_auc_roc
        
    def Find_Optimal_Cutoff(self, target, predicted):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations

        predicted : Matrix with predicted data, where rows are observations

        Returns
        -------     
        list type, with optimal cutoff value

        """
        fpr, tpr, threshold = roc_curve(target, predicted)
        
        i = np.arange(len(tpr)) 
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
        self.df_roc = roc
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
           
        return list(roc_t['threshold']) 
        
    def get_test_set_all(self):
        return self.__lst_test_set_all
    
    def get__emb_dict(self):
        return self.__emb_dict
    
    def get__df_nodes(self):
        return self.__df_nodes
    
    def get__df_merge(self):
        return self.__df_merge
    
    def get_the_model(self):
        return self.logreg
    
    def get_emb_labels(self):
        self.__label_all_node_embedding()
        return self.__labeled_emb
    
    def get_node_type_to_id(self):
        return self.node_type_to_id
        
    def __read_embeddings(self, mp2v_emb = True):

        filepath = self.__data_path + self.__emb_file + '.txt'
        self.__emb_dict = dict([])
        chk_list = []
        with open(filepath) as fp:
            line = fp.readline().split(' ')
            print('__read_embeddings| row: {}, col:{}'.format(line[0], line[1]))
            line = fp.readline()
            cnt = 0
            while line:
                
                strLine = line.strip().split(' ')
                if strLine[0] == '</s>':
                    line = fp.readline()
                    continue
                    
                if mp2v_emb :
                    entity_id = int(strLine[0][1:])
                else:
                    entity_id = int(strLine[0])
                
                vector = strLine[1:]                
                vec = [float(i) for i in vector]
                self.__emb_dict[entity_id] = vec
                cnt += 1
                
                chk_list.append(entity_id)
                line = fp.readline()
        
        print('__read_embeddings| size of emb: {}, count line: {}, list chk: {}'.format(len(self.__emb_dict), cnt, len(chk_list) ))
        print()
