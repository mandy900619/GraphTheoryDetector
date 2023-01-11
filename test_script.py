import os

models = ['rf','mlp','knn']

dirname = './TestingBin/'

# malware detection
for model in models:
    for bin in os.listdir(dirname):
        cmd = 'python -W ignore main.py -i ' + dirname+bin \
            + ' -m ' + model \
            + ' -o ./TestingResult/GraphTheoryDetector_'+model+'_MD.csv'
        os.system(cmd)

# family classification
for model in models:
    for bin in os.listdir(dirname):
        cmd = 'python -W ignore main.py -i ' + dirname+bin \
            + ' -m ' + model \
            + ' -o ./TestingResult/GraphTheoryDetector_'+model+'_FC.csv' \
            + ' -c'
        os.system(cmd)