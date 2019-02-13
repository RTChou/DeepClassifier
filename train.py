import argparse

def main():
    parser = argparse.ArgumentParser(description="This script is for training the semi-supervised neural network model")
    parser.add_argument('-i', '--input', required=True, help='training gene expression data, with genes in rows and samples in columns')
    parser.add_argument('-o', '--output', required=True, help='output filename')
    args = parser.parse_args()

