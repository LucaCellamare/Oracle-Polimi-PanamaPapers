#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:21:49 2019

@author: aparravi
"""

import pandas as pd
import time
from sklearn.model_selection import train_test_split
import numpy as np
import ray
import psutil

from evaluation_script import read_file, recall_at_k, precision_at_k

#############################
#############################

def timeit(method):
    """
    Standard Python decorator that measures the execution time of a method;
    """
    def timed(*args, **kw):
        start = time.time()
        result = method(*args, **kw)
        end = time.time()
        
        print(f"{method.__name__}:  {(end - start):.2f} s")
        return result
    return timed

#############################
#############################

@timeit
def load_training_data(data_path: str, row_subset: float=1, train_split: float=0.7, shuffle: bool=False, seed=None):
    """
    Load the training set and divide it into training and test splits.
    "LinkedID" is the value that we want to predict;

    :param data_path: path to the dataset to load;
    :param row_subset: use only the specified fraction of rows in the dataset (value in (0, 1]);
    :param train_split: fraction of rows placed in the training set;
    :param shuffle: if True, shuffle the rows before splitting or subsetting the data;
    """ 
    if row_subset <= 0 or row_subset > 1:
        row_subset = 1

    data = read_file(training_file, set_record_id_as_index=True)
    if shuffle:
        data = data.sample(frac=1, random_state=seed)
    # Obtain the specified subset of rows;
    data = data.iloc[:int(np.ceil(len(data) * row_subset))]
        
    X = data.drop(columns="linked_id")
    y = data["linked_id"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split, shuffle=shuffle, random_state =seed)

    return X_train, X_test, y_train, y_test

    #############################
    #############################

@timeit
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform cleaning of the dataset, e.g. lowercase strings, fill missing values, etc...
    """
    cleaned_data = data.copy()
    for c in ["name", "type", "address", "phone", "email", "modification"]:
        cleaned_data[c] = cleaned_data[c].str.lower()

    return cleaned_data

#############################
#############################

@timeit
def preprocess_phone(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform cleaning of the dataset, e.g. lowercase strings, fill missing values, etc...
    """
    cleaned_data = data.copy()
    phone=cleaned_data["phone"]
    phone_list=list(phone)
    #phone_max=phone.max()
    #print("max phone length :")
    #print(phone_max)
    #for c in ["name", "type", "address", "phone", "email", "modification"]:
    #    cleaned_data[c] = cleaned_data[c].str.lower()

    return cleaned_data

#############################
#############################  
@timeit
def index_data(data: list) -> list:
    """
    Manipulate the data to create indices that speed up the computation. 

    The dataframes to be indexed are passed with a list of pd.DataFrames.
    Be careful not to leak information from the train dataset to the test dataset!

    In this case, replace all strings with unique integer values,
    so that comparing rows for equality is much faster;
    """

    # Obtain the original ending of each dataframe.
    # Prepend a 0 to indicate the start of the first dataframe;
    lengths = [0] + [len(x) for x in data]
    lengths = np.cumsum(lengths)
    # Join together all the input data;
    concatenated_data = pd.concat(data)
        
    for c in ["name", "type", "address", "phone", "email", "modification"]:
        concatenated_data[c] = pd.factorize(concatenated_data[c])[0]

    # Split the input data;
    indexed_data = []
    for i in range(len(lengths) - 1):
        indexed_data += [concatenated_data.iloc[lengths[i]:lengths[i + 1], :]]

    return indexed_data

#############################
#############################

def predict_record(X_train: pd.DataFrame, y_train: pd.Series, record: pd.DataFrame, K: int=10) -> list:
    """
    Given a record and a training set, find the records in the training set that best match
    the input record. Predictions can be done using very simple unsupervised algorithms,
    as done in this example, or computed with some fancy ML model;
    """
    # In this simple model, look for the number of identical columns for each training record.
    # Skip the last column (type of modification);
    identical_values_per_row = (record[:-1] == X_train.iloc[:, :-1]).sum(axis=1)
    # Obtain the K rows with the most matches;
    best_matches = identical_values_per_row.sort_values(ascending=False)[:K]
    # Retrieve the original record IDs from the training set;
    return list(y_train.loc[best_matches.index.values])

@timeit
def predict_er(X_train, y_train, X_test, K=10) -> dict:
    """
    Given a training dataset and a test dataset, obtain the top-K predictions 
    for each record in the test dataset;
    """

    # Store for each record the list of predictions;
    predictions = {}
    start = time.time()

    # Compute a prediction for each record;
    for i, (record_id, row_i) in enumerate(X_test.iterrows()):
        if i % 1000 == 0 and i > 0:
            print(f"processed {i}/{len(X_test)} records, {100 * i / len(X_test):.2f}%")
            print(f"\ttime elapsed: {(time.time() - start):.2f} s")
        # Find the best matching record for the current record;
        predictions[record_id] = predict_record(X_train, y_train, row_i, K)
    return predictions

#############################
#############################

def predict_record_fast(X_train: np.array, y_train: pd.Series, record: np.array, K: int=10) -> list:
    """
    Given a record and a training set, find the records in the training set that best match
    the input record. Predictions can be done using very simple unsupervised algorithms,
    as done in this example, or computed with some fancy ML model;
    """
    # In this simple model, look for the number of identical columns for each training record.
    # Skip the last column (type of modification);
    identical_values_per_row = pd.Series(np.logical_and(record == X_train, record >= 0, X_train >= 0).sum(axis=1))
    # Obtain the K rows with the most matches;
    best_matches = identical_values_per_row.sort_values(ascending=False)[:K]    
    # Retrieve the original record IDs from the training set;
    return list(y_train.loc[best_matches.index.values])

@timeit
def predict_er_fast(X_train, y_train, X_test, K=10) -> dict:
    """
    Given a training dataset and a test dataset, obtain the top-K predictions 
    for each record in the test dataset;
    """

    # Store for each record the list of predictions;
    predictions = {}
    start = time.time()

    # Extract the matrix that represent the data.
    # Skip the last column;
    X_train_matrix = X_train.values[:, :-1]

    # Also reindex the y column, as we lose the original index when doing a comparison using matrices;
    y_train_vector = y_train.reset_index(drop=True)

    # Compute a prediction for each record;
    for i, (record_id, row_i) in enumerate(X_test.iterrows()):
        # Extract values from the current row;
        row_i_vector = row_i[:-1].values
        if i % 1000 == 0 and i > 0:
            print(f"processed {i}/{len(X_test)} records, {100 * i / len(X_test):.2f}%")
            print(f"\ttime elapsed: {(time.time() - start):.2f} s")
        # Find the best matching record for the current record;
        predictions[record_id] = predict_record_fast(X_train_matrix, y_train_vector, row_i_vector, K)
    return predictions

#############################
#############################

# Ray: 
# - https://github.com/ray-project/ray
# - https://towardsdatascience.com/modern-parallel-and-distributed-python-a-quick-tutorial-on-ray-99f8d70369b8

@ray.remote
def predict_record_parallel(record_id: str, X_train: pd.DataFrame, y_train: pd.Series, record: pd.DataFrame, K: int=10) -> list:
    """
    Given a record and a training set, find the records in the training set that best match
    the input record. Predictions can be done using very simple unsupervised algorithms,
    as done in this example, or computed with some fancy ML model;
    """
    # In this simple model, look for the number of identical columns for each training record.
    # Skip the last column (type of modification);
    identical_values_per_row = pd.Series(np.logical_and(record == X_train, record >= 0, X_train >= 0).sum(axis=1))
    # Obtain the K rows with the most matches;
    best_matches = identical_values_per_row.sort_values(ascending=False)[:K]
    # Retrieve the original record IDs from the training set;
    return {"id": record_id, "predictions": list(y_train.loc[best_matches.index.values])}

@timeit
def predict_er_parallel(X_train, y_train, X_test, K=10) -> dict:
    """
    Given a training dataset and a test dataset, obtain the top-K predictions 
    for each record in the test dataset;
    """

    # Store for each record the list of predictions;
    predictions_temp = []
    start = time.time()

    # Extract the matrix that represent the data.
    # Skip the last column;
    X_train_matrix = X_train.values[:, :-1]
    # Also reindex the y column, as we lose the original index when doing a comparison using matrices;
    y_train_vector = y_train.reset_index(drop=True)
    # Store the values in the shared memory space of Ray;
    X_train_r = ray.put(X_train_matrix)
    y_train_r = ray.put(y_train_vector)

    # Compute a prediction for each record;
    for i, (record_id, row_i) in enumerate(X_test.iterrows()):
        # Extract values from the current row;
        row_i_vector = row_i[:-1].values
        if i % 1000 == 0 and i > 0:
            print(f"processed {i}/{len(X_test)} records, {100 * i / len(X_test):.2f}%")
            print(f"\ttime elapsed: {(time.time() - start):.2f} s")
        # Find the best matching record for the current record;
        predictions_temp += [predict_record_parallel.remote(record_id, X_train_r, y_train_r, row_i_vector, K)]
    predictions = ray.get(predictions_temp)
        
    return {p["id"]: p["predictions"] for p in predictions}
                
#############################
#############################

@timeit
def prediction_dict_to_df(predictions: dict) -> pd.DataFrame:
    # Turn the prediction dict into a series of tuples;
    results = []
    for query_id, pred_list in predictions.items():
        for p in pred_list:
            results += [[query_id, p]]
    return pd.DataFrame(results, columns=["queried_record_id", "predicted_record_id"])

@timeit
def prediction_dict_to_kaggle_df(predictions: dict) -> pd.DataFrame:
    # Turn the prediction dict into a series of tuples;
    results = []
    for query_id, pred_list in predictions.items():
        results += [[query_id, " ".join(pred_list)]]
    return pd.DataFrame(results, columns=["queried_record_id", "predicted_record_id"])

@timeit
def kaggle_sol_to_df(kaggle_df: pd.DataFrame) -> pd.DataFrame:
    kaggle_df_indexed = kaggle_df.set_index("queried_record_id")
    results = []
    for query_id, pred_list in kaggle_df_indexed.iterrows():
        results += [[query_id, pred] for pred in pred_list["predicted_record_id"].split(" ")]
    return pd.DataFrame(results, columns=["queried_record_id", "predicted_record_id"])

#############################
#############################
    
#%%

if __name__ == "__main__":

    # Startup of Ray;
    num_cpus = psutil.cpu_count(logical=False)
    ray.shutdown()
    ray.init(num_cpus=num_cpus)

    #%% 1. Build the dataset;
    training_file = "/Users/marti/Downloads/oracle-polimi-contest-2019/data/entity-resolution_advanced-topics-training_data.csv"
    X_train, X_test, y_train, y_test = load_training_data(training_file, shuffle=True, row_subset=0.01, seed=42)



    #print(X_test)
    #print(X_train)
    #print(y_train) 
    #print(y_test)

    #%% 2. Clean data;
    X_train = clean_data(X_train)
    X_test = clean_data(X_test)
    #X_train=preprocess_phone(X_train)
    #X_test=preprocess_phone(X_test)

    X_train, X_test = index_data([X_train, X_test])

    #print(X_test)
    #print(X_train)
    #print(y_train) 
    #print(y_test)
    # Recombine the datasets, to easily compute metrics;
    train = X_train.merge(y_train, left_index=True, right_index=True)
    test = X_test.merge(y_test, left_index=True, right_index=True)

    #%% 4. Compute the predictions;
    #predictions = predict_er(X_train, y_train, X_test)
    predictions = predict_er_fast(X_train, y_train, X_test)
    #predictions = predict_er_parallel(X_train, y_train, X_test)

    # Rewrite predictions in a format compatible with our metrics;
    pred_df = prediction_dict_to_df(predictions)

    #%% 5. Compute recall@K;
    recall_dict = recall_at_k(pred_df, train, test)
    print(recall_dict["AverageRecall"])
    print(recall_dict["AverageFilteredRecall"])

    #%% 6. Compute MAP@K;
    precision_dict = precision_at_k(pred_df, train, test)
    print(precision_dict["AveragePrecision"])
    print(precision_dict["AverageFilteredPrecision"])

    #%% 7. Store predictions;
    # This is useful only for debugging, we are doing predictions on a subset of the training set,
    # not on the real test set!
    pred_df.to_csv("/Users/marti/Downloads/oracle-polimi-contest-2019/data/predictions_parra_test.csv", index=False)

    # Write the solution with a Kaggle-compatible format;
    pred_df_kaggle = prediction_dict_to_kaggle_df(predictions)
    pred_df_kaggle.to_csv("/Users/marti/Downloads/oracle-polimi-contest-2019/data/predictions_parra_test_kaggle.csv", index=False)

'''
    print(X_test)
    print(X_train)
    print(y_train) 
    print(y_test)

    maxColumnLenghts = []
    for col in range(len(X_train.columns)):
        maxColumnLenghts.append(min(X_train.iloc[:,col].astype(str).apply(len)))
    print('Max Column Lengths ', maxColumnLenghts)

    numberInteractions=X_train.shape[0]
    index_list=list(X_train.index.values)
    numRecords=len(index_list)

    #X_train['record_id'].nunique()
    print(numberInteractions)
    #numRecords=list(set(numRecords))

    print('num of total records :'+str(numRecords))
    numRecords_unique=len(list(set(index_list)))
    print('num of total unique records :'+str(numRecords_unique))
    #num_total_names=X_train['name']

    numNames=X_train['name'].nunique()
    #idList_unique = list(set(X_train[0]))
    #nameList_unique = list(set(X_train[1]))

    #numRecords = len(idList_unique)
    #numNames = len(nameList_unique)


    print ("Number of names\t {}, Number of records\t {}".format(numNames, numRecords_unique))
    #print ("Max ID items\t {}, Max Id users\t {}\n".format(max(nameList_unique), max(idList_unique)))
    print ("Average interactions per user {:.2f}".format(numberInteractions/numRecords_unique))
    print ("Average interactions per item {:.2f}\n".format(numberInteractions/numNames))

    print ("Sparsity {:.2f} %".format((1-float(numberInteractions)/(numberInteractions*6))*100))
'''
    #%% 3. Create indices for the data;
    # If doing indexing, use "predict_er_fast" or "predict_er_parallel" in this example;
