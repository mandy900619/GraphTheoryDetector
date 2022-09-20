# GraphTheoryDetector - FeatureConstruction

## Description

* The aim of this folder is generate the training data from gpickle which represents the FCG of binary file.
* **input.** a folder which contains gpickle
* **output.** a csv file for feature vector

## Files

* **gpickle2feature.py** : main program for feature construction
* **graphity.py** : a module for extraction
* **param_parser.py** : for parsing the args

## Usage

e.g.

```
python gpickle2feature.py --input-path ../Benign_FCG --output-path ../Dataset
```

