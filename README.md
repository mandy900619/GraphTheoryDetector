# Graph Theory Detector

> ref. ALASMARY, Hisham, et al. Analyzing and detecting emerging Internet of Things malware: A graph-based approach. *IEEE Internet of Things Journal*, 2019, 6.5: 8977-8988.

## Introduction

### Description

* The main program is a malware detector.

  * **Input** : a binary file

  * **Output** : its label predicted by ML model
  * **Flow** : 
    * reverse the bin and extract the feature
    * load the model
    * predict


### Feature Extraction

* We reverse the binary file to function call graph(FCG) by r2pipe, then extract the attribute of FCG:

  * number of nodes

  * number of edges

  * density

  * (mean, max, min, median, std) of closeness_centrality


  * (mean, max, min, median, std) of betweenness_centrality


  * (mean, max, min, median, std) of degree_centrality


  * (mean, max, min, median, std) of shortestpaths.avglen 

  The dimension of feature is 23.


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

* **FeatureConstruction** : some code about gpickle to feature.csv (for training, validation)
* **Modeling** : some code about training and saving the model
* **MD(FC)_Model** : save the model with .joblib, e.g. 'rf.joblib'
* **TestingBin** : some binary for testing
* **main.py** : the detector(classifier)

- **graphity.py** : module for feature extraction
- **param_parser.py** : for parsing args
- **scaler.joblib** : scale the feature vector

## Usage

* setting path of input file

  ```
  python main.py --input-path [FILE_PATH]
  ```

* select the model

  ```
  python main.py --model [MODEL]
  ```

  MODEL can be rf, knn, svm, mlp, default: mlp

* if you wanna do family classification

  ```
  python main.py --MDorFC FC
  ```

* e.g.

  ```
  python -W ignore main.py --input-path .\TestingBin\1100a1693fbe43b0ff29c1d5a36011421752e162171683a7053cc1a342cdb11a --model svm --MDorFC FC
  ```

  * ignore the warning message by using '-W ignore'



## Result of training

|    Models    | Accuracy | Precision | Recall | F1_score | Time cost |
| :----------: | :------: | :-------: | :----: | :------: | :-------: |
| RandomForest |  0.965   |   0.968   | 0.965  |  0.965   |   0.64    |
|     KNN      |  0.953   |   0.953   | 0.953  |  0.953   |   6.01    |
|     SVM      |  0.824   |   0.827   | 0.822  |  0.823   |   41.22   |
|     MLP      |  0.871   |   0.871   | 0.871  |  0.871   |   0.66    |
