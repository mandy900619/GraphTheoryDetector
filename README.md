# Graph Theory Detector

> ref. ALASMARY, Hisham, et al. Analyzing and detecting emerging Internet of Things malware: A graph-based approach. *IEEE Internet of Things Journal*, 2019, 6.5: 8977-8988.

## Introduction

### Description

* Input a binary file
* Output its label predicted by ML model

### Feature Construction

* generate FCG

* extract the attribute of graph:

  1. number of nodes
  2. number of edges
  3. density
  4. -8. (mean, max, min, median, std) of closeness_centrality

  9. -13. (mean, max, min, median, std) of betweenness_centrality

  14. -18. (mean, max, min, median, std) of degree_centrality

  19. -23. (mean, max, min, median, std) of shortestpaths.avglen 


## Requirements

* python3
* radare2
* python package
  * r2pipe
  * networkx
  * joblib
  * argparse
  * sklearn

## Files

* **FeatureConstruction** : some code about gpickle to feature.csv
* **Modeling** : some code about training and saving the model
* **MD(FC)_Model** : save the model with .joblib, e.g. 'rf.joblib'
* **TestingBin** : some binary for testing
* **main.py** : the detector(classifier)

- **graphity.py** : module for feature extraction
- **param_parser.py** : for parsing args
- **scaler.joblib** : 

## Usage

* setting path of input file

  ```
  python main.py --input-path [FILE_PATH]
  ```

* select the model

  ```
  python main.py --model [MODEL]
  ```

  MODEL can be rf, knn, svm, mlp

* e.g.

  ```
  python -W ignore main.py --input-path .\test_data\1100a1693fbe43b0ff29c1d5a36011421752e162171683a7053cc1a342cdb11a --model svm
  ```

  * ignore the warning message by using '-W ignore'



## Result of training

|    Models    | Accuracy | Precision | Recall | F1_score | Time cost |
| :----------: | :------: | :-------: | :----: | :------: | :-------: |
| RandomForest |  0.965   |   0.968   | 0.965  |  0.965   |   0.64    |
|     KNN      |  0.953   |   0.953   | 0.953  |  0.953   |   6.01    |
|     SVM      |  0.824   |   0.827   | 0.822  |  0.823   |   41.22   |
|     MLP      |  0.871   |   0.871   | 0.871  |  0.871   |   0.66    |
