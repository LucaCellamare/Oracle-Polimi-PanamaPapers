#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:02:47 2019

@author: aparravi
"""

import ray
import psutil

from evaluation_script import read_file
from test_er import clean_data, index_data, \
    predict_er_parallel, \
    prediction_dict_to_kaggle_df

if __name__ == "__main__":

    # Startup of Ray;
    num_cpus = psutil.cpu_count(logical=False)
    ray.shutdown()
    ray.init(num_cpus=num_cpus)
    
    #%% 1. Build the dataset;
    training_file = "/Users/marti/Downloads/oracle-polimi-contest-2019/data/entity-resolution_advanced-topics-training_data.csv"
    train = read_file(training_file, set_record_id_as_index=True)
        
    X_train = train.drop(columns="linked_id")
    y_train = train["linked_id"]
    
    test_file = "/Users/marti/Downloads/oracle-polimi-contest-2019/data/test_data.csv"
    X_test = read_file(test_file, set_record_id_as_index=True)
       
    #%% 2. Clean data;
    X_train = clean_data(X_train)
    X_test = clean_data(X_test)
    
    #%% 3. Create indices for the data;
    X_train, X_test = index_data([X_train, X_test])
        
    #%% 4. Compute the predictions;
    predictions = predict_er_parallel(X_train, y_train, X_test)
    
    # Write the solution with a Kaggle-compatible format;
    pred_df_kaggle = prediction_dict_to_kaggle_df(predictions)
    pred_df_kaggle.to_csv("/Users/marti/Downloads/oracle-polimi-contest-2019/data/predictions_test_kaggle.csv", index=False)