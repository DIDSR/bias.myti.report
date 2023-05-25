import argparse
import os
import pandas as pd

def ensemble_results():
    """
    function to load scores from individual models, concat them and compute mean score, and finally output the mean score
    """
    main_dir = args.main_dir
    model_number = args.model_number

    temp = []
    # load scores from models
    for RD in range(model_number):
        prediction = pd.read_csv(os.path.join(main_dir, f"{args.folder_name}_{RD}", args.prediction_file))
        prediction = prediction.rename(columns={'Yes':f"Yes score {RD}"})
        predictions = prediction[f"Yes score {RD}"]        
        temp.append(predictions)
    
    #compute average scores
    ensemble_scores = pd.concat(temp, axis=1, ignore_index=False)
    ensemble_scores = ensemble_scores.loc[:,~ensemble_scores.columns.duplicated()]
    print(ensemble_scores.head(8))
    cols = [f"Yes score {RD}" for RD in range(model_number)]
    ensemble_scores["Yes"] = ensemble_scores.loc[:, cols].mean(axis = 1)
    
    #output the average score to a new file
    patient_id = prediction['patient_id'].copy()
    ensembleResults = ensemble_scores['Yes'].copy()
    result = pd.concat([patient_id, ensembleResults], axis=1)
    result.to_csv(os.path.join(main_dir, f"{args.folder_name}_0", args.output_file),index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_dir',type=str)
    parser.add_argument('--model_number',type=int, default=10)
    parser.add_argument('--folder_name',type=str)
    parser.add_argument('--prediction_file',type=str)
    parser.add_argument('--output_file',type=str)
    args = parser.parse_args()
    ensemble_results()
