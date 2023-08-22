import os
import argparse
import traceback
import pandas as pd
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Extraction function
"""
code reference
https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
"""
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame

    Parameters
    ----------
    path : str
        path to tensorflow log file

    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interest', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--tb_path', type=str, default='./tensorboard')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_path = args.save_path
    tb_path = args.tb_path
    
    os.makedirs(save_path, exist_ok=True)
    
    interest = args.interest
    
    folders = [os.path.join(tb_path, folder) for folder in os.listdir(tb_path) if interest in folder]
    
    metric_keys = ['Recall', 'Specificity', 'F1-Score', 'Test Accuracy', 'Test Loss', 'AUROC', 'Test Accuracy (Balanced)', 'Train Loss']
    
    metrics_dict = defaultdict(list)
    
    for folder in folders:
        file = os.listdir(folder)
        df = tflog2pandas(f'{folder}/{file[0]}')
        for k in metric_keys:
            metrics_dict[k] = df.loc[df['metric']==k]['value']
        df = pd.DataFrame.from_dict(metrics_dict)
        file_name = result = '_'.join(folder.split('_')[2:])    # I use YYYY-MM-DD_HHMMSS_{interest}_{details} for tb event name so...
        df.to_csv(f"{save_path}/{file_name}.csv")
        print(f"{save_path}/{file_name}.csv SAVED!")
        
        
    result_path=save_path
    desired_stages = ['SITTING', '1', '2', '3', '4', 'all', '#1', '#2', '#3', 'resting']
    total_files = [os.path.join(result_path, file) for file in os.listdir(result_path) if file.endswith('.csv')]
    keywords = []
    files = []
    
    for stage in desired_stages:
        try:
            keywords.append('_'.join((interest, stage)))
        except:
            print(f"NO FILE IN {file}")
    for file in total_files:
        for keyword in keywords:
            if keyword in file:
                files.append(file)
    
    final_dict = {stage:{} for stage in desired_stages}
    
    for file in files:
        df = pd.read_csv(file)
        stage = file.split('_')[-1].split('.')[0]
        for key in metric_keys:
            if key == 'Recall':
                final_dict[stage]['Sensiticity'] = df[key].to_numpy()[-1]
            elif key == 'Test Accuracy':
                final_dict[stage][key] = df[key].to_numpy()[-1]/100
            else:
                final_dict[stage][key] = df[key].to_numpy()[-1]
                
    final_df = pd.DataFrame.from_dict(final_dict)
    final_df = final_df[desired_stages]
    os.makedirs('./result_summaries', exist_ok=True)
    final_df.to_csv(f'./result_summaries/{interest}.csv')
    print(f'{interest}.csv SAVED!')