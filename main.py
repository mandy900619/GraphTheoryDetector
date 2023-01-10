import graphity
import numpy as np
import networkx as nx
import r2pipe
from joblib import load

from utils import parameter_parser
from utils import write_output

def bin2fcg(path):
    '''
    param: binary file path
    return: a string describe the FCG in .dot form
    '''

    # OpenFile
    r2 = r2pipe.open(path)
    # Analyze  
    r2.cmd('aaaa')
    # Extract and return the dot string 
    command = 'agCd'
    G = r2.cmd(command)
    return G

def Extract_feature(dotString):
    '''
    param: a string describe the FCG in .dot form
    return: feature vector of the sample
    '''

    feature=[]
    #Create graph
    G=graphity.create_graph(dotString)

    # append #nodes & # edges
    feature.append(G.number_of_nodes())
    feature.append(G.number_of_edges())

    # append Density
    feature.append(graphity.get_density(G))

    # append Closeness Centrality
    for i in graphity.closeness_centrality(G):
        feature.append(i)

    # append Betweeness Centrality
    for i in graphity.betweeness_centrality(G):
        feature.append(i)

    # append Degree Centrality
    for i in graphity.degree_centrality(G):
        feature.append(i)

    # append Shortest Path
    for i in graphity.shortest_path(G):
        feature.append(i)

    return feature

def Scaling(feature):
    '''
    param: feature vector
    return: feature vector after scaling
    '''
    scaler = load('scaler.joblib')
    feature = scaler.transform(feature)
    return feature

def Predict(X,clf,is_family_classification):
    '''
    param: X (feature vector)
    return: y (label), 0 for benign, 1 for malware
    '''
    if is_family_classification:
        model = load('./FC_Model/'+clf+'.joblib')
    else:
        model = load('./MD_Model/'+clf+'.joblib')

    result = model.predict_proba(X).tolist()[0]
    return result

def main(args):
    result = [-1] # default: fail to predict -> -1
    
    # init the labels
    labels = ['BenignWare','Malware']
    if args.classify:
        labels = ['BenignWare', 'Mirai', 'Bashlite', 'Unknown', 'Android', 'Tsunami', 'Dofloo', 'Xorddos', 'Hajime', 'Pnscan']

    # extract FCG from binary
    try:
        dotString = bin2fcg(args.input_path)
    except:
        print('fail to extract the FCG.')
        print(result)
        write_output(args.input_path, args.output_path, result, labels)
        return result
    
    # extract feature vector from FCG
    try:
        feature = Extract_feature(dotString)
    except:
        print('fail to extract the feature.')
        print(result)
        write_output(args.input_path, args.output_path, result, labels)
        return result

    # prediction
    feature = np.array(feature).reshape(1,-1)
    feature = Scaling(feature)  
    result = Predict(feature,args.model,args.classify)
    
    # output
    print(result)
    write_output(args.input_path, args.output_path, result, labels)
    return result


if __name__=='__main__':
    args = parameter_parser()
    main(args)