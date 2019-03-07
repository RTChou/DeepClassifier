import argparse
from keras.models import load_model
import pickle

def main():
    parser = argparse.ArgumentParser(description="This script is for predicting using the trained semi-supervised neural network model")
    parser.add_argument('-e', '--exp', required=True, help='path to input gene expression data, with genes in rows and samples in columns')
    parser.add_argument('-m', '--model', required=True, help='path to input trained model')
    parser.add_argument('-b', '--label_bin', required=True, help='path to input label binarizer')

    args = parser.parse_args()

if __name__ == "__main__":
    main()

