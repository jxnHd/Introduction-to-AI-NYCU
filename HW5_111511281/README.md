Introduction to AI 2025 HW5 - Graph Neural Network

Overview

This project implements a GraphSAGE model to classify user nationalities in the Treads social network dataset. The implementation uses PyTorch Geometric.

Environment Setup

Install the required dependencies:

pip install -r requirements.txt


How to Run

Place Dataset:
Ensure the following files are in the same directory:

train.csv

test.csv

treads_graph.csv

run python3 main.py for starting the training

Train the GraphSAGE model.

Validate on a 10% split.

Save the best model to {studentID}.pth.

Generate predictions in {studentID}.csv.



Model Architecture

Type: GraphSAGE (Graph Sample and Aggregate)

Layers: 3

Hidden Dimensions: 512

Regularization: Batch Normalization, Dropout (p=0.5)

Optimizer: AdamW

References

PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/

Inductive Representation Learning on Large Graphs (Hamilton et al., 2017)
