import pandas as pd
import os

def load_data(dir, amp):
  """ Loads all of the tsv files in the directory into a single tsv (with variable indicating the source tsv) """
  df_list = []
  for file in os.listdir(dir):
    if file.endswith(".tsv"):
      h = [0,1] if amp == 'indirect' else [0]
      temp_df = pd.read_csv(os.path.join(dir, file), sep='\t', index_col=[0,1], header=h)
      temp_df = temp_df.reset_index().rename(columns={'level_0':'details', 'level_1':'mean/std'})
      temp_df = temp_df.melt(id_vars=['details', 'mean/std'])
      temp_df = temp_df.pivot(index=[c for c in temp_df.columns if c not in ['mean/std', 'value']], columns='mean/std', values='value').reset_index()
      if amp == 'direct':
        temp_df = temp_df.rename(columns={'variable':'dist'})
        if "B" in temp_df['dist'].values[0]:
          subgroups = ["B", "W"]
        else:
          subgroups = ['F', "M"]
        prev_diff = {}
        PA = {}
        for d in temp_df['dist'].unique():
          val = int(d[:-2])
          prev_diff[d] = abs(50 - val)*2
          if val > 50:
            PA[d] = subgroups[0]
          elif val < 50:
            PA[d] = subgroups[1]
          else:
            PA[d] = 'N/A'
        temp_df["Training Prevalence Difference"] = temp_df['dist'].map(prev_diff)
        temp_df["Positive-associated"] = temp_df['dist'].map(PA) 
      temp_df['mitigation'] = file.replace('.tsv','')
      df_list.append(temp_df)
  return pd.concat(df_list)
  
def unpack_details(df, details:dict):
  convert = {x:{} for x in details.keys()}
  for D in df.details.unique():
    for det, options in details.items():
      for o in options:
        if o in D:
          convert[det][D] = o
          break
  for k in convert.keys():
    df[k] = df['details'].map(convert[k])
  df['task'] = 'COVID'
  df.loc[df['details'] == 'AUROC sex','task'] = 'Subgroup'
  df.loc[df['details'] == 'Overall AUROC (Sex)','task'] = 'Subgroup'
  return df
  
def clean_data(df, **kwargs):
  for k, d in kwargs.items():
    df[k] = df[k].map(d)
  return df
  

    
if __name__ == "__main__":
  test_dir = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/Bias_manipulation_manuscript/results/indirect-race/"
  df = load_data(test_dir)
  details = {"metric":['AUROC', 'Sensitivity', 'Prevalence']}
  details['subgroup'] = ["Overall"] + [x for x in df['Positive-associated'].unique() if x != 'N/A']
  df = unpack_details(df, details)
  df = clean_data(df, mitigation={'baseline':'No Mitigation', 'reject_object_classification':'Reject Option Classification', 'image_cropping':'Image Cropping', 'reweighing':"Reweighing", 'calibrated_equalized_odds':"Calibrated Equalized Odds"})
  print(df)
  #print(df.details.unique())