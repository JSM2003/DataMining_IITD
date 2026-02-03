# DataMining_IITD


## Assignment 1

#### 1) Frequent Itemset Mining

#### 2) Frequent Subgraph Mining

#### 3) Graph Indexing
3.1: Folder Structure 
``` 
./
├── convert.sh
├── direct_si.py
├── env.sh
├── generate_candidates.sh
├── identify.sh
└── q3.pdf 
```
3.2 Build and Run
```
bash env.sh #to build environment
```

```
bash identify.sh <path_graph_dataset> <path_discriminative_subgraphs>

# <path_graph_dataset> : absolute filepath to the dataset of database graphs.
# <path_discriminative_subgraphs>: absolute filepath to store the discriminative
subgraphs
```
```
bash convert.sh <path_graphs> <path_discriminative_subgraphs> <path_features>

# <path_graphs>: absolute filepath to the dataset of graphs. These can be database
graphs or query graphs.
# <path_discriminative_subgraphs>: absolute filepath to the discriminative
subgraphs.
# <path_features>: absolute filepath to store the input dataset mapped to the
feature space. This must be a 2D numpy array.
```
```
bash generate_candidates.sh <path_database_graph_features>
<path_query_graph_features> <path_out_file>

# <path_database_graphs>: absolute filepath to the 2D numpy array of database
graphs dataset.
# <path_query_graphs>: absolute filepath to the 2D numpy array of query graphs
dataset.
# <path_out_file>: absolute filepath to the file for storing candidate sets for
the query graphs.
```