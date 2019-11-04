import pandas as pd
import argparse
import time
import numpy as np

def read_file(path: str, set_record_id_as_index: bool=False) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, escapechar="\\", index_col="record_id" if set_record_id_as_index else None)

def recall_at_k(resultTable : pd.DataFrame, trainingData: pd.DataFrame, testingData: pd.DataFrame) -> dict:
    """
    Given a list of K predictions for each query, first retrieve the correct ID from the test data,
    then look in the training data the percentage of records that have been successfully identified.
    
    For example, given query "1234-M", first retrieve the correct ID "1234" from the test data,
    then obtain from the training data all records that refer to "1234", 
    and finally look how many of them we have found;
    """
    
    # Obtain all the predictions for each record in the test set;
    perQueryRecords = resultTable.groupby("queried_record_id")
    
    # Group training records by their LinkedID truth value;
    groupedTrainingRecords = trainingData.groupby("linked_id")

    totalRecall = 0.0

    allRecords = dict()
    
    start = time.time()
    for i, (queriedRecordID, group) in enumerate(perQueryRecords):
        if i % 1000 == 0 and i > 0:
            print(f"processed {i}/{len(perQueryRecords)} records, {100 * i / len(perQueryRecords):.2f}%")
            print(f"\tcurrent recall: {(totalRecall / i):.2f}")
            print(f"\ttime elapsed: {(time.time() - start):.2f} s")
        
        try:
            queriedLinkedID = testingData.loc[queriedRecordID, "linked_id"]
        except IndexError:
            raise IndexError(f"ID {queriedRecordID} not found in testing data!")
        
        try:
            allRelevantRecords = set(groupedTrainingRecords.get_group(queriedLinkedID).index.values)
        except KeyError:
            allRelevantRecords = set()
        setPredictedRecords = set(group["predicted_record_id"])
        selectedRelevantRecords = setPredictedRecords.intersection(allRelevantRecords)
        recall = 1
        if (len(allRelevantRecords) > 0):
            recall = len(selectedRelevantRecords) / len(allRelevantRecords)

        totalRecall += recall
        allRecords[queriedRecordID] = [queriedRecordID, recall, len(selectedRelevantRecords), len(allRelevantRecords)]
    
    # Store the results in a summary table;
    result_table =  pd.DataFrame.from_dict(
                        allRecords,
                        orient='index',
                        columns=["QueriedRecordID", "Recall@K", "SelectedRecords", "AllRelevantRecords"]
                    )
    # Compute the filtered recall, which considers only queries with at least one relevant record in the training data;
    queries_with_relevant_records = result_table[result_table["AllRelevantRecords"] > 0]
    filtered_recall = np.mean(queries_with_relevant_records["SelectedRecords"] / queries_with_relevant_records["AllRelevantRecords"])

    return {
            "AverageRecall" : totalRecall / len(perQueryRecords),
            "AverageFilteredRecall": filtered_recall,
            "perQueryResult" : result_table
            }
    
def precision_at_k(resultTable : pd.DataFrame, trainingData: pd.DataFrame, testingData: pd.DataFrame) -> dict:
    """
    Given a list of K predictions for each query, first retrieve the correct ID from the test data,
    then look in the training data the percentage of records that are actually relevant;
    
    For example, given query "1234-M", first retrieve the correct ID "1234" from the test data,
    then obtain from the training data all records that refer to "1234", 
    and finally look how many of the records we have found are actually referring to "1234"
    """
    
    # Obtain all the predictions for each record in the test set;
    perQueryRecords = resultTable.groupby("queried_record_id")
    
    # Group training records by their LinkedID truth value;
    groupedTrainingRecords = trainingData.groupby("linked_id")

    totalPrecision = 0.0
    numberOfPredictionsForRelevantRecords = 0

    allRecords = dict()
    
    start = time.time()
    for i, (queriedRecordID, group) in enumerate(perQueryRecords):
        if i % 1000 == 0 and i > 0:
            print(f"processed {i}/{len(perQueryRecords)} records, {100 * i / len(perQueryRecords):.2f}%")
            print(f"\tcurrent precision: {(totalPrecision / i):.2f}")
            print(f"\ttime elapsed: {(time.time() - start):.2f} s")
        
        try:
            queriedLinkedID = testingData.loc[queriedRecordID, "linked_id"]
        except IndexError:
            raise IndexError(f"ID {queriedRecordID} not found in testing data!")
        
        try:
            allRelevantRecords = set(groupedTrainingRecords.get_group(queriedLinkedID).index.values)
        except KeyError:
            allRelevantRecords = set()
        setPredictedRecords = set(group["predicted_record_id"])
        selectedRelevantRecords = setPredictedRecords.intersection(allRelevantRecords)
        precision = 1
        if (len(allRelevantRecords) > 0):
            precision = len(selectedRelevantRecords) / len(setPredictedRecords)
            numberOfPredictionsForRelevantRecords += len(setPredictedRecords)

        totalPrecision += precision
        allRecords[queriedRecordID] = [queriedRecordID, precision, len(selectedRelevantRecords), len(allRelevantRecords)]
    
    # Store the results in a summary table;
    result_table =  pd.DataFrame.from_dict(
                        allRecords,
                        orient='index',
                        columns=["QueriedRecordID", "Precision@K", "SelectedRecords", "AllRelevantRecords"]
                    )
    # Compute the filtered recall, which considers only queries with at least one relevant record in the training data;
    queries_with_relevant_records = result_table[result_table["AllRelevantRecords"] > 0]
    filtered_precision = np.mean(queries_with_relevant_records["SelectedRecords"] / numberOfPredictionsForRelevantRecords)

    return {
            "AveragePrecision" : totalPrecision / len(perQueryRecords),
            "AverageFilteredPrecision": filtered_precision,
            "perQueryResult" : result_table
            }   

#%%

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compute the Recall@K value for the input predictions.")
    parser.add_argument("-p", "--predictions", metavar="<path/to/predictions.csv>",
                        help="Path to a file with predictions from your model.")
    parser.add_argument("-k", "--kaggle_format", action="store_true",
                        help="Specify if predictions are in kaggle format.")
    parser.add_argument("-t", "--training_data", metavar="<path/to/training.csv>",
                        help="Path to a training data file.", default="/Users/marti/Downloads/oracle-polimi-contest-2019/data/entity-resolution_advanced-topics-training_data.csv")
    parser.add_argument("-s", "--testing_data", metavar="<path/to/test.csv>",
                        help="Path to a testing data file.", default="/Users/marti/Downloads/oracle-polimi-contest-2019/data/entity-resolution_advanced-topics-testing_data.csv")   
    parser.add_argument("-o", "--output_path", metavar="<path/to/output.csv>",
                        help="Path to a file where the results per individual query should be stored.",
                        default="/Users/marti/Downloads/oracle-polimi-contest-2019/data/evaluation_output.csv")   

    args = parser.parse_args(["-p", "/Users/marti/Downloads/oracle-polimi-contest-2019/data/predictions_test_kaggle.csv", "-o", "/Users/Downloads/oracle-polimi-contest-2019/data/test_output.csv", "-k"])
    resultFile = args.predictions
    trainingFile = args.training_data
    testingFile = args.testing_data
    outputFile = args.output_path
    kaggleFormat = args.kaggle_format
    
    resultTable  = read_file(resultFile)
    # Turn predictions into a format compatible with the scoring metrics, if they are provided in Kaggle format;
    if kaggleFormat:
        # Defer import to avoid circular dependencies from files;
        from test_er import kaggle_sol_to_df
        resultTable = kaggle_sol_to_df(resultTable) 
    trainingData = read_file(trainingFile, set_record_id_as_index=True)
    testingData  = read_file(testingFile, set_record_id_as_index=True)
        
    recall_dict = recall_at_k(resultTable, trainingData, testingData)
    print(f"recall: {recall_dict['AverageRecall']:.3f}")
    print(f"filtered recall: {recall_dict['AverageFilteredRecall']:.3f}")
    # On the sample dataset, we get 0.623 and 0.323
    
    precision_dict = precision_at_k(resultTable, trainingData, testingData)
    print(f"precision: {precision_dict['AveragePrecision']:.3f}")
    print(f"filtered precision: {precision_dict['AverageFilteredPrecision']:.3f}")
    # On the sample dataset, we get 0.507 and 0.000
    
#    recall_dict["perQueryResult"].to_csv(outputFile, index = False)
