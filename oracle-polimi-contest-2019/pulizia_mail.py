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
def load_training_data(data_path: str, row_subset: float=1, train_split: float=1, shuffle: bool=False, seed=None):
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
        
    for c in ["name", "address", "phone", 'mail']:
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
def imputing(a):
	return 
def foo_bar(a):
	d=a.split(',')
	c = Counter(d)
	numero_missing=d.count('missing')  
	mode_count = max(c.values())
	if mode_count!=numero_missing:
		mode = {key for key, count in c.items() if count == mode_count}
		return list(mode)[0]
	else:
		cleaned_list = [ x for x in d if x is not 'missing']
		if len(cleaned_list)==0:
			#h=[]
			#h.append('missing')
			return -1
		else:
			e = Counter(cleaned_list)
			mode_count2 = max(e.values())
			g = {key for key, count in e.items() if count == mode_count2}
		
	return list(g)[0]
#############################
#############################


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

	#mail_col=X_train['email']
	my_list_train = X_train["email"].values
	my_list_test= X_test["email"].values
	mail_domain_train=[]

	for mail in my_list_train:
		if type(mail) is not float:
			if len(mail.split('@'))==2:
				mail_domain_train.append('@' + mail.split('@')[1])
			elif len(mail.split('@'))>2:
				mail_split=mail.split('@')[-1:]
				mail_domain_train.append(mail_split)
			else:
				mail_domain_train.append(-1)
		else:
			mail_domain_train.append(-1)

	#print(mail_domain)
	#print("la len dei domain è: ")
	#print(len(mail_domain))
	#print("\n\n")
	mail_extension_train=[]
	for mail in mail_domain_train:
		if mail is not -1:
			if len(mail.split('.'))==2:
				mail_extension_train.append(mail.split('.')[1])
			elif len(mail.split('.'))>2:
				mail_split=mail.split('.')[-1:]
				mail_extension_train.append(mail_split[0])
			else:
				mail_extension_train.append(-1)

		else:
			mail_extension_train.append(-1)

	#print(mail_extension)	
	#print(len(mail_extension))
	#mapping = dict(enumerate(mail_domain))

	#mapping2 = dict(enumerate(mail_extension))
	mail_domain_test=[]
	for mail in my_list_test:
		if type(mail) is not float:
			if len(mail.split('@'))==2:
				mail_domain_test.append('@' + mail.split('@')[1])
			elif len(mail.split('@'))>2:
				mail_split=mail.split('@')[-1:]
				mail_domain_test.append(mail_split)
			else:
				mail_domain_test.append(-1)
		else:
			mail_domain_test.append(-1)

	#print(mail_domain)
	#print("la len dei domain è: ")
	#print(len(mail_domain))
	#print("\n\n")
	mail_extension_test=[]
	for mail in mail_domain_test:
		if mail is not -1:
			if len(mail.split('.'))==2:
				mail_extension_test.append(mail.split('.')[1])
			elif len(mail.split('.'))>2:
				mail_split=mail.split('.')[-1:]
				mail_extension_test.append(mail_split[0])
			else:
				mail_extension_test.append(-1)

		else:
			mail_extension_test.append(-1)

	X_train['mail_domain'] = mail_domain_train

	X_train['mail_extension'] = mail_extension_train
	X_test['mail_domain'] = mail_domain_test
	X_test['mail_extension'] = mail_extension_test
	true_mail_extension=[]
	X_train=X_train[["name","mail_extension","mail_domain","phone","address","email"]]
	#X_train=X_train[["name","mail_extension"]]
	#X_test=X_test[["name","mail_extension","mail_domain","phone","address","email"]]
	X_train=X_train.fillna('missing')
	X_train=X_train.replace(-1,"missing")
	#foo = lambda a: "/".join(a) 
	df_new = X_train.groupby('name').agg(','.join)
	df_new = df_new.applymap(foo_bar)
	df_new=df_new.replace(['missing'],-1)
	df_new.mask(df_new==-1, inplace=True)
	#df_new['mail_extension'] = mode(df_new['mail_extension'].split(','))
	#df_new=df_new[['name','phone','modification','mail_extension','type','address']]
	print(df_new[0:10])
	print("len df_new è :")
	print(df_new.shape)
	#X_train.merge(df_new, left_on='name', right_on='name')[X_train.name,]
	#X_train.loc[X_train.name.isin(df_new['name'])]
	#cols = ['name']
	#d2 = df_new.set_index('name')
	#X_train.replace(-1,X_test.drop('Expiry', 1).join(d2, on=cols))
	#X_train['mail_extension'].fillna(X_train.drop('mail_extension', 1).join(d2, on='name'))
	#merged_df['visit_date'] = pd.to_datetime(merged_df['visit_date']) 
	#date_info_df['calendar_date'] = pd.to_datetime(date_info_df['calendar_date']) 


	#s = date_info_df.set_index('calendar_date')['day_of_week']
	#t = date_info_df.set_index('day_of_week')['holiday_flg']

	#merged_df['day_of_week'] = merged_df['day_of_week'].fillna(merged_df['visit_date'].map(s))
	#merged_df['holiday_flg'] = merged_df['holiday_flg'].fillna(merged_df['day_of_week'].map(t))


	#X_train_unique=X_train.drop_duplicates('name')
	X_train_not_unique = X_train[X_train.duplicated(subset=['name'], keep=False)]
	# this line is optional (depending on the layout of data.)
	#merged_df.visit_date = pd.to_datetime(merged_df.visit_date)

	# do replacement here.
	mapping=df_new.set_index(["name"])
	X_train["mail_extension"] = X_train.name.map(mapping["mail_extension"])
	X_train["phone"] = X_train.name.map(mapping["phone"])
	print(X_train[0:10])


	#X_train.fillna(df1.drop('', 1).join(d2, on=cols))
	#%% 3. Create indices for the data;
	# If doing indexing, use "predict_er_fast" or "predict_er_parallel" in this example;

'''
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
	pred_df.to_csv("/Users/marti/Downloads/oracle-polimi-contest-2019/data/predictions_plus_domain_extension_test.csv", index=False)

	# Write the solution with a Kaggle-compatible format;
	pred_df_kaggle = prediction_dict_to_kaggle_df(predictions)
	pred_df_kaggle.to_csv("/Users/marti/Downloads/oracle-polimi-contest-2019/data/predictions_plus_domain_extension_test_kaggle.csv", index=False)
'''
