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
    * write the result -> output csv file


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
* input binary: `-i <path>`, `--input-path <path>`
* model: `-m <model>`, `--model <model>`
* output (record): `-o <path>`, `--output-path <path>`
* Malware Detection / Family Classification
    * if you wanna do malware detection(binary clf) -> do nothing
    * else if (wanna do family classification) -> `-c`
* e.g.
    `python -i testingBin/0021eaf2 -o myDetector_FC_records.csv -m rf -c`
    * using trained rf family classifier(`-c`), predict '0021eaf2' and write the result to 'myDetector_FC_records.csv'

* e.g.

  `python -W ignore main.py -i .\TestingBin\1100a1693fbe43b0ff29c1d5a36011421752e162171683a7053cc1a342cdb11a -o GraphTheoryDetector_FC_Test.csv -m mlp -c`

  * ignore the warning message by using '-W ignore'



## Result of Malware Detection

|    Models    | Accuracy | Time cost |
| :----------: | :------: | :-------: |
| RandomForest |   0.97   |   5.01    |
|     KNN      |  0.967   |  165.51   |
|     SVM      |  0.772   |  1606.21  |
|     MLP      |  0.879   |   49.51   |

## Result of Family Classification

|    Models    | Accuracy | Time cost |
| :----------: | :------: | :-------: |
| RandomForest |  0.952   |   4.21    |
|     KNN      |  0.944   |  104.92   |
|     SVM      |   0.71   |  1562.06  |
|     MLP      |  0.887   |   70.99   |
