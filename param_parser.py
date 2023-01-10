import argparse


# def parameter_parser():
#     parser = argparse.ArgumentParser(description="Run GraphTheoryDetector.")

#     parser.add_argument('--input-path',
#                         nargs='?',
#                         default='./TestingBin/0000dc2f3c8bde2d3b61cd1ba3aa5e839c0a7bf432d2e06a88a7ce3b199453e7',
#                         help='input binary file.')

#     parser.add_argument('--model',
#                         nargs='?',
#                         default='mlp',
#                         help='Select the model(rf, knn, svm, mlp).')
    
#     parser.add_argument('--MDorFC',
#                         nargs='?',
#                         default='MD',
#                         help='if it is expected to classify the family of malware, type \'--MDorFC FC\', then the program will import the FC model.')

#     return parser.parse_args()

def parameter_parser():
    # update the description with your detector name
    parser = argparse.ArgumentParser(description="Run Detector.")

    parser.add_argument('-i', '--input-path', type=str, metavar='<path>',
                        help='path to the binary file')

    parser.add_argument('-c', '--classify', action='store_true',
                        help='apply the family classifier')

    parser.add_argument('-m', '--model', type=str, metavar='[ rf | knn | svm | mlp ]', default='mlp',
                        help='model to predict')

    parser.add_argument('-o', '--output-path', type=str, metavar='<path>', 
                        help='path to the output file')
    args = parser.parse_args()
    return args