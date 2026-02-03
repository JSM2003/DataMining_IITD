# DataMining_IITD


## Assignment 1

#### 1) Frequent Itemset Mining
1.1: Folder Structure 
``` 
./
├── q1_1.sh
├── q1_2.sh
├── q1.pdf
├── generate_dataset.py
├── plot.py
```
1.2: Run
```
bash q1_1.sh <path_apriori_executable> <path_fp_executable> <path_dataset>
<path_out>

# <path_apriori_executable>: absolute filepath to apriori's compiled
executable
# <path_fp_executable>: absolute filepath to fp-tree's compiled executable
# <path_dataset>: absolute filepath to the dataset file
# <path_out>: absolute folderpath where the plot and the outputs at different
thresholds will be saved
``` 
```
bash q1_2.sh <universal_itemset> <num_transactions>

# <universal_itemset>: superset of distinct items possible across all
transactions
# <num_transactions>: number of transactions in the dataset
```
#### 2) Frequent Subgraph Mining
2.1: Folder Structure
```
./
├── graph_lib_binaries/
│   ├── fsg*
│   ├── gaston*
│   └── gSpan*
├── plot_scripts/
│   └── results_plot.py
├── preproc_scripts/
│   ├── fsg_data_adaptor.py
│   └── gspan_gaston_data_adaptor.py
├── q2.sh*
├── raw_dataset/
│   └── yeast_167.txt_graph
└── q2.pdf
```

2.2: Run
```
bash q2.sh <path_gspan_executable> <path_fsg_executable>
<path_gaston_executable> <path_dataset> <path_out>

# <path_gspan_executable>: absolute filepath to gspan's compiled executable
# <path_fsg_executable>: aboslute filepath to fsg's compiled executable
# <path_gaston_executable>: aboslute filepath to gaston's compiled executable
# <path_dataset>: absolute filepath to the dataset file
# <path_out>: absolute folderpath where the plot and the outputs at different
minimum supports will be saved
```

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