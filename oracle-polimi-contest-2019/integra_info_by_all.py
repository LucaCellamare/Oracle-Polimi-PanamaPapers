import pandas as pd
import time
from sklearn.model_selection import train_test_split
import numpy as np
import ray
import psutil
import re
from evaluation_script import read_file, recall_at_k, precision_at_k
import statistics 
from statistics import mode
from collections import Counter
#import tqdm

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
    for c in ["name","type","address", "phone", "email", "modification"]:
        cleaned_data[c] = cleaned_data[c].str.lower()
    
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
        
    for c in ["name", "address", "phone", "mail_extension"]:
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

def foo_bar(a):
    d=a.split(',')
    
    c = Counter(d)
    #numero_missing=d.count('missing')  
    mode_count = max(c.values())
    #if mode_count!=numero_missing:
    mode = {key for key, count in c.items() if count == mode_count}
    return list(mode)
@timeit
def func(X_train,df_new,cols):
    X_train_pulito=X_train.copy()
    for row_index,row in X_train_pulito.iterrows():
        for j in range(len(cols)):
            val=X_train_pulito.get_value(row_index,cols[j])
            if val=='missing':
                X_train_pulito.set_value(row_index,cols[j],df_new[cols[j]].loc[X_train_pulito.name[row_index]])
    return X_train_pulito
@timeit
def func2(X_train,df_new,cols):
    X_train_pulito=X_train.copy()
    for row_index,row in X_train_pulito.iterrows():
        for j in range(len(cols)):
            val=X_train_pulito.get_value(row_index,cols[j])
            if val=='missing':
                X_train_pulito.set_value(row_index,cols[j],df_new[cols[j]].loc[X_train_pulito.phone[row_index]])
    return X_train_pulito


@timeit
def func3(X_train,df_new,cols):
    X_train_pulito=X_train.copy()
    for row_index,row in X_train_pulito.iterrows():
        for j in range(len(cols)):
            val=X_train_pulito.get_value(row_index,cols[j])
            if val=='missing':
                X_train_pulito.set_value(row_index,cols[j],df_new[cols[j]].loc[X_train_pulito.address[row_index]])
    return X_train_pulito

@timeit
def func4(X_train,df_new,cols):
    X_train_pulito=X_train.copy()
    for row_index,row in X_train_pulito.iterrows():
        for j in range(len(cols)):
            val=X_train_pulito.get_value(row_index,cols[j])
            if val=='missing':
                X_train_pulito.set_value(row_index,cols[j],df_new[cols[j]].loc[X_train_pulito.email[row_index]])
    return X_train_pulito
if __name__ == "__main__":
    
    # Startup of Ray;
    num_cpus = psutil.cpu_count(logical=False)
    ray.shutdown()
    ray.init(num_cpus=num_cpus)

    #%% 1. Build the dataset;	
    training_file = "/Users/marti/Downloads/oracle-polimi-contest-2019/data/entity-resolution_advanced-topics-training_data.csv"
    X_train, X_test, y_train, y_test = load_training_data(training_file, shuffle=True, seed=42)

    X_train = clean_data(X_train)
    X_test = clean_data(X_test)
    X_train['record_id']=X_train.index
    X_train=X_train.fillna('missing')
    X_train=X_train.replace(-1,"missing")
    n_m_n=X_train.loc[ (X_train['address'])=="missing", 'address' ].shape[0]
    n_m_a=X_train.loc[ (X_train['phone'])=="missing", 'phone' ].shape[0]
    n_m_e=X_train.loc[ (X_train['email'])=="missing", 'email' ].shape[0]
    print(f"There are {n_m_n} missing-values in column address")
    print(f"There are {n_m_a} missing-values in column phone")
    print(f"There are {n_m_e} missing-values in column email")
    df_new=X_train.copy()
    df_new = df_new.groupby('name').agg(','.join)
    df_new = df_new.applymap(foo_bar)
    df_new=df_new.sort_values('name')
    df_new=df_new[['phone','address','email']]
    
    cols=df_new.columns
    X_train=X_train[['name','phone','address','email']]
    X_train_pulito=func(X_train,df_new,cols)
    X_train_pulito=X_train_pulito.applymap(lambda x : 'missing' if x==['missing'] else x)
    X_train_pulito=X_train_pulito.applymap(lambda x: [y for y in x if y!="missing"] if isinstance(x, list) else x)

    df_new2=X_train.copy()
    df_new2 = df_new2.groupby('phone').agg(','.join)
    df_new2 = df_new2.applymap(foo_bar)
    df_new2=df_new2.sort_values('phone')
    df_new2=df_new2[['name','address','email']]
    
    cols2=df_new2.columns
    X_train_pulito=X_train_pulito[['phone','name','address','email']]
    X_train_pulito2=func2(X_train_pulito,df_new2,cols2)
    X_train_pulito2=X_train_pulito2.applymap(lambda x : 'missing' if x==['missing'] else x)
    X_train_pulito2=X_train_pulito2.applymap(lambda x: [y for y in x if y!="missing"] if isinstance(x, list) else x)
    
    df_new3=X_train.copy()
    df_new3 = df_new3.groupby('address').agg(','.join)
    df_new3 = df_new3.applymap(foo_bar)
    df_new3=df_new3.sort_values('address')
    df_new3=df_new3[['name','phone','email']]
    
    cols3=df_new3.columns
    X_train_pulito2=X_train_pulito2[['address','name','phone','email']]
    X_train_pulito3=func3(X_train_pulito2,df_new2,cols3)
    X_train_pulito3=X_train_pulito3.applymap(lambda x : 'missing' if x==['missing'] else x)
    X_train_pulito3=X_train_pulito3.applymap(lambda x: [y for y in x if y!="missing"] if isinstance(x, list) else x)
    
    df_new4=X_train.copy()
    df_new4 = df_new4.groupby('email').agg(','.join)
    df_new4 = df_new4.applymap(foo_bar)
    df_new4=df_new4.sort_values('email')
    df_new4=df_new4[['name','phone','address']]
    
    cols4=df_new4.columns
    X_train_pulito3=X_train_pulito3[['email','name','phone','address']]
    X_train_pulito4=func2(X_train_pulito3,df_new2,cols4)


    X_train_pulito4=X_train_pulito4.applymap(lambda x : -1 if x=='missing' else x)
    X_train_pulito4=X_train_pulito4.applymap(lambda x: [y for y in x if y!="missing"] if isinstance(x, list) else x)
    print(X_train_pulito4.phone[0:10])
    print(X_train_pulito4.address[0:10])
    print(X_train_pulito4.email[0:10])

    num_miss_name = X_train_pulito4.loc[ (X_train_pulito4['address'])==-1, 'address' ].shape[0]
    num_miss_address = X_train_pulito4.loc[X_train_pulito4['phone']==-1 , 'phone' ].shape[0]
    num_miss_email = X_train_pulito4.loc[X_train_pulito4['email']==-1 , 'email' ].shape[0]

    print(f"There are {num_miss_name} missing-values in column address")
    print(f"There are {num_miss_address} missing-values in column phone")
    print(f"There are {num_miss_email} missing-values in column email")