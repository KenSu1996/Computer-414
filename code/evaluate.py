import os
import random
import torch
import pickle
import math
import sklearn
import numpy as np
import pandas as pd
# from utils import those function
from utils import precision,recall,f1_score,confusion_matrix, roc_curve, det_curve, accuracy

# Settings and configuration
model_state = "Pretrained"
predictions_file = "results/pred.csv"
results_dir = "results"

# Function to round down a number to a specified number of decimals
def round_down(number, decimals):
    multiplier = 10 ** decimals
    return math.floor(number * multiplier) / multiplier

# Function to process and prepare data for evaluation
def process_data():
    torch.backends.cudnn.benchmark = True
    print("Loading Data")
    predictions_df = pd.read_csv(predictions_file)
    fpr_group, tpr_group, thresholds_group = roc_curve(predictions_df['label'], predictions_df['VGGFace2'])
    fpr_group_det, fnr_group_det, thresholds_group_det = det_curve(predictions_df['label'], predictions_df['VGGFace2'])

    print("Processing Data")
    threshold_values = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    verification_accuracy = []
    accuracy_dict = {}

    for threshold in threshold_values:
        for fpr, tpr in zip(fpr_group, tpr_group):
            if round_down(fpr, len(str(threshold))-2) == threshold:
                verification_accuracy.append(tpr)
                accuracy_dict[threshold] = tpr
                break

    print(verification_accuracy, flush=True)
    with open(f'{results_dir}/verification_accuracy_all_{model_state}.pkl', 'wb') as file:
        pickle.dump(accuracy_dict, file)

    process_attributes(predictions_df, 'e1', 'ethnicity')
    process_attributes(predictions_df, 'g1', 'gender')
    process_attributes(predictions_df, 'a1', 'attributes')

def process_attributes(dataframe, attribute_col, attribute_name):
    attribute_dict = {}
    attribute_specific_dict = {}

    for value in dataframe[attribute_col].unique():
        temp_df = dataframe.loc[dataframe[attribute_col] == value]
        fpr_group_det, fnr_group_det, thresholds_group_det = det_curve(temp_df['label'], temp_df['VGGFace2'])
        fpr_group, tpr_group, thresholds_group = roc_curve(temp_df['label'], temp_df['VGGFace2'])

        accuracy_for_threshold = {}
        for threshold in [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for fpr, tpr in zip(fpr_group, tpr_group):
                if round_down(fpr, len(str(threshold))-2) == threshold:
                    accuracy_for_threshold[threshold] = tpr
                    break

        attribute_dict[value] = accuracy_for_threshold
        attribute_specific_dict[value] = accuracy_for_threshold[0.01]

    print(attribute_dict, flush=True)
    with open(f'{results_dir}/verification_{attribute_name}_{model_state}.pkl', 'wb') as file:
        pickle.dump(attribute_specific_dict, file)
    with open(f'{results_dir}/verification_{attribute_name}_all_{model_state}.pkl', 'wb') as file:
        pickle.dump(attribute_dict, file)

if __name__ == '__main__':
    process_data()
