<!-- Made By Nathan Gravel -->
# RegPattern2Vec 

  

## Introduction  

  

  

  

This repository contains the code and datasets for the manuscript "Predicting Protein and Pathway Associations for Understudied Dark Kinases using Pattern-constrained Knowledge Graph Embedding ".    

  

  

#### Included in this repository are the following:  

  

  - `DownloadLinkForData.txt`: file (the onedrive link in the README.md is the same in this text file )

    - file contains link for download  

    - Download contains KG, embeddings, uniprot2genename, and reactome2pathwayname files  
    
    - This also contains files that goes with the example provided below  
  
   - `data/GP2VEC_emb_walk_file_40_40_0_0_bwV4_s256_w7_n5_mc5.txt`: embedding KG file
   
   - `data/GP2VEC_emb_walk_file_40_40_0_0_bwV4_s256_w7_n5_mc5`: embedding KG file   

   - `data/df_edges.csv`: main KG edge file 

   - `data/df_node.csv`: KG node file 

   - `data/df_relation.csv`: KG relation type file 

   - `data/df_type.csv`: KG node type file 

   - `data/original/GeneNameMapping.txt`: UniProtID to name mapping file. One of the input files for GeneratingKinaseGraph 

   - `data/original/ReactomePathways.txt`: ReactomeID to name mapping file. One of the input files for GeneratingKianseGraph 
  
   - `benchmarkCofigFiles`: Has 3 files that can be used to set the paramaters for the supplementary benchmarks models

   

- 2 Jupyter lab notebooks files 

  - Notebook for measuring the accuracy `graphpattern2vec_process-multithread-Edited2022.ipynb` 

  - Notebook for prediction generation `graphpattern2vec_process-multithread-PREDICTION-Edited2022.ipynb` 

   

- `gp2v.yml`: file 

  - file can be used to create the environment  

   

- `graphpattern2vec`: folders  

  - folder holds the objects/functions of RegPattern2Vec 

  

- `Readme.md`: file 

  - You're reading it right now 

   

- `model`: folder  

  - folder holds temporary files  

 

   

## Installing dependencies   

  

![python=3.10.8](https://img.shields.io/badge/Python-3.10.8-green) 

Python == 3.10.8 

  

From pip: 

  

![jupyterlab=3.5.0](https://img.shields.io/pypi/v/jupyterlab?label=jupyterlab) 

![numpy=1.23.5](https://img.shields.io/pypi/v/numpy?label=numpy) 

![pandas=1.5.2](https://img.shields.io/pypi/v/pandas?label=pandas) 

![matplotlib=3.6.2](https://img.shields.io/pypi/v/matplotlib?label=matplotlib) 

![scikit-learn=1.1.3](https://img.shields.io/pypi/v/scikit-learn?label=scikit-learn) 

![statsmodels=0.13.5](https://img.shields.io/badge/statsmodels-0.13.5-blue) 

![tqdm=4.64.1](https://img.shields.io/badge/tqdm-4.64.1-blue) 

![networkx==2.8.8](https://img.shields.io/pypi/v/networkx?label=networkx) 

  

 

  

### optional environment  

![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white) 

  

  

## Downloading this repository  

```  

git clone https://github.com/gravelCompBio/RegPattern2Vec.git

```  

```  

cd RegPattern2vec/   

```  

### ALSO DOWNLOAD THIS data/ FOLDER AND PUT IT IN THE RegPattern2Vec-main folder (unzip and make sure the name of the unzipped folder is still "data") 

  

https://outlookuga-my.sharepoint.com/:u:/g/personal/nmg60976_uga_edu/EatErUI0YUNMnDxxk-LOcnYBvG30lkW2weNIv2WuUfZVzw?e=Ig92AU 

or (same data differnt download locations)

DOI:

10.5281/zenodo.7541827


  

  

  

## Installing dependencies with conda ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white) 

the yml file is in RegPatter2Vec file and is named "gp2v.yml"   

```  

conda env create -f gp2v.yml -n gp2v 

```  

```  

conda activate gp2v 

```  

  

## Utilizing the Model  

### downloading needed file 

  

1) Confirm prerequisite data has been downloaded (see download repository section) 

2) Navigate to RegPattern2vec-main folder in the terminal and run jupyter lab  

  

```  

jupyter lab 

```  

#### once in the jupyter lab navigate to either  

#### ------ graphpattern2vec_process-multithread-Edited2022   


- for measuring accuracy of the model (AUC-ROC)  

- has split funtion for benchmarking so only will not give the best predictions

  

or 

#### ------ graphpattern2vec_process-multithread-PREDICTION-Edited2022.ipynb   

- for generating predictions   

- ROC in this file does not represent the overall accuracy  

  

  

## Please Read this Section before running the link prediction section in either notebook!  Section for generating user embeddings from outside data!   

  

1)  Confirm prerequisite data has been downloaded (see download repository section) 

 

  

2) After performing the random walk sections of the code in either jupyter notebook, either the embedding files provided can be used or embedding files can be generated from the user’s own data.  If you wish to generate your own embedding files, please see the section below for how to do so with metapath2vec.   

#### A) If the user creates their own embedding please note the current version of the code assumes that you are just following the example provided and the use of our provided embedding (generated with our data and provided Knowledge Graph).  Users will 	need to change file name in code (both notebooks). 

  

3) After embedding file generation,  run the link prediction sections of the code in either notebook 

  

### Basic guide for running metapath2vec to generate new embedding files from user generated Random Walk file  


#### This is only if you want to run this pipeline on custom data. If you are running this pipline on our data/KG provided, you do not have to run metapath2vec because the we provide the embedding files (output of metapath2vec) in the `DownloadLinkForData.txt` 

Also the version of metapath2vec we use (https://github.com/Change2vec/change2vec/tree/master/metapath2vec/code_metapath2vec ) is computationally expensive and our KG took 200Gb of ram with over 2 days of runtime with 64 cores so if you can find a more effencint version of metapath2vec that you trust, I would suggest you use others versions (one example is embiggen(grape) https://github.com/AnacletoLAB/grape)

We utilize files produced by our model containing nodes collected through RegPattern2Vec as input into a portion of the pipeline for metapath2vec used vectorize the KG nodes and generate embeddings  

If the user is generating their own data and using their own embedding files, all cells in the notebook should be run until the Link Prediction section then you should create the embeddings with metapath2vec. 

 
## metapath2vec and createing the embeddings   
The code can be accessed from:    

#### https://github.com/Change2vec/change2vec/tree/master/metapath2vec/code_metapath2vec 

The following parameters were used to produce our embeddings: 

- Size 256 

- Window 7 

- Negative 5 

- MinCount 5 

- Threads 32 

- PlusPlus 1 

  

Example code of how we put it in the terminal  

  

  

  

```  

make metapath2vec -B

```

``` 
./metapath2vec -train data/walk_GP2vec_test_REL_simplified_sep_probs_40_40_0_0_bwV4.csv -output data/GP2VEC_emb_walk_file_40_40_0_0_bwV4_s256_w7_n5_mc5 -threads 32 -size 256 -window 7 -negative 5 -min-count 5 -pp 1
``` 

 Run the rest of the cells after the Link Prediction section, after inputting embedding file generated outside of notebook. 

 

see following paper for a more complete explanation  

#### Dong, Y., Chawla, N. V., & Swami, A. (2017, August). metapath2vec: Scalable representation learning for heterogeneous networks. In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 135-144). 

  

  

 

## Model Output  

After running the Link Prediction Notebook (data/results/df_prediction_name_emb_walk_file_40_40_0_0_bwV4_s256_w7_n5_mc5.csv 

) , the model will output a CSV file: 

Example here with first 5 rows: 
``` 
uniprot,ProteinName,Pathway_id,Pathway_Name,Score 

Q9H8X2,IPPK,1855191,Synthesis of IPs in the nucleus,0.999999998802044 

Q96GX5,MASTL,2465910,MASTL Facilitates Mitotic Progression,0.9999999956283079 

Q9H8X2,IPPK,1855167,Synthesis of pyrophosphates in the cytosol,0.9999999950700171 

Q13237,PRKG2,8978934,Metabolism of cofactors,0.9999999891668171 

Q13237,PRKG2,1474151,"Tetrahydrobiopterin (BH4) synthesis, recycling, salvage and regulation",0.9999999819057269 
``` 
 

Row Names: 

Uniprot 

Gene Name 

Pathway ID 

Pathway_Name 

Score 

 

 
