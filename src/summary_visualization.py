import seaborn
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
import os

# formatting
colors = seaborn.color_palette("Paired")
rcParams['font.family'] = 'monospace'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 14
hfont = {'fontname':'monospace', 'fontweight':'bold', 'fontsize':18}
rcParams['figure.figsize'] = (8,6)


# summary json file locations
# RICORD_1c_json_path = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/summary_table_RICORD_1c.json"
RICORD_1c_json_path = "/gpfs_projects/ravi.samala/OUT/2022_CXR/summary_table__MIDRC_RICORD_1C.json"
open_AI_json_path = "/gpfs_projects/ravi.samala/OUT/2022_CXR/summary_table__open_AI_20patients.json"
COVID_19_AR_json_path = "/gpfs_projects/ravi.samala/OUT/2022_CXR/summary_table__COVID_19_AR.json"
# designate repo(s) to summarize
# repo_list = ['RICORD-1c', 'open_AI', 'COVID_19_AR']
repo_list = ['COVID_19_AR']

# set save location
# save_loc = "/gpfs_projects/alexis.burgon/OUT/2022_CXR"
save_loc = "/gpfs_projects/ravi.samala/OUT/2022_CXR/figs/"

# different characteristic summaries
def summarize_sex(data, dataframe, repo):
    for row in dataframe['patient_info']:
        if 'sex' in row[0]:
            if row[0]['sex'] == 'F':
                data.loc[repo]['Female'] += 1
            elif row[0]['sex'] == 'M':
                data.loc[repo]['Male'] += 1
            else:
                data.loc[repo]['Not Specified'] += 1
        else:
            data.loc[repo]['Not Specified'] += 1
    return data

def summarize_COVID(data, dataframe, repo):
    for row in dataframe['patient_info']:
        if 'COVID_positive' in row[0]:
            if row[0]['COVID_positive'] == 'Yes':
                data.loc[repo]['Positive'] += 1
            elif row[0]['COVID_positive'] == 'No':
                data.loc[repo]['Negative'] += 1
            else:
                data.loc[repo]['Not Specified'] += 1
        else:
            data.loc[repo]['Not Specified'] += 1
    return data

def summarize_race(data, dataframe, repo):
    for row in dataframe['patient_info']:
        if not 'race' in row[0]:
            data.loc[repo]['Not Specified'] += 1
        elif row[0]['race'] == 'Missing':
            data.loc[repo]['Not Specified'] += 1
        elif row[0]['race'] not in data.columns:
            data[str(row[0]['race'])] = np.zeros((data.shape[0],1))
            data.loc[repo][row[0]['race']] += 1
        else:
            data.loc[repo][row[0]['race']] += 1
    return data

def summarize_ethnicity(data, dataframe, repo):
    for row in dataframe['patient_info']:
        if not 'ethnicity' in row[0]:
            data.loc[repo]['Not Specified'] += 1
        elif row[0]['ethnicity'] == 'Missing':
            data.loc[repo]['Not Specified'] += 1
        elif row[0]['ethnicity'] not in data.columns:
            data[str(row[0]['ethnicity'])] = np.zeros((data.shape[0],1))
            data.loc[repo][row[0]['ethnicity']] += 1
        else:
            data.loc[repo][row[0]['ethnicity']] += 1
    return data

def summarize_age(data, dataframe, repo):
    for row in dataframe['patient_info']:
        if not 'age' in row[0]:
            data.loc[repo]['Not Specified'] += 1
        elif row[0]['age'] == 'Missing':
            data.loc[repo]['Not Specified'] += 1
        elif str(row[0]['age']) not in data.columns:
            data[str(row[0]['age'])] = np.zeros((data.shape[0],1))
            data.loc[repo][str(row[0]['age'])] += 1
        else:
            data.loc[repo][str(row[0]['age'])] += 1
    return data

def summarize_modality(data, dataframe, repo):
    num_patients = 0
    num_images = 0
    for row in dataframe['images_info']:
        patient_flag = False
        for entry in row:
            if 'modality' in entry:
                if entry['modality'] == 'CR':
                    data.loc[repo]['CR'] += 1
                    patient_flag = True
                    num_images += 1
                elif entry['modality'] == 'DX':
                    data.loc[repo]['DX'] += 1
                    patient_flag = True
                    num_images += 1
                else:
                    data.loc[repo]['Not Specified'] += 1
            else:
                data.loc[repo]['Not Specified'] += 1
        if patient_flag:
            num_patients += 1
    return num_patients, num_images, data

def summarize_body_part(data, dataframe, repo):
    for row in dataframe['images_info']:
        for entry in row:
            if 'body part examined' not in entry:
                dataframe.loc[repo]['Not Specified'] += 1
            elif entry['body part examined'] == 'Missing':
                dataframe.loc[repo]['Not Specified'] += 1
            elif entry['body part examined'] not in data.columns:
                data[entry['body part examined']] = np.zeros((data.shape[0],1))
                data.loc[repo][entry['body part examined']] += 1
            else:
                data.loc[repo][entry['body part examined']] += 1

    return data

# utility
def pie_label(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)

# different types of outputs
def pie_chart(data, save_name, include_numbers = True, fig_title=None):
    f, ax = plt.subplots(figsize=(8,6))
    data = {i:j for i,j in data.items() if j != 0}
    if include_numbers:
        plt.pie(list(data.values()), labels=list(data.keys()), colors=colors,autopct=lambda pct: pie_label(pct, list(data.values())))
    else:
        plt.pie(list(data.values()), labels=list(data.keys()), colors=colors,autopct='%.0f%%')
    if fig_title:
        plt.title(fig_title, **hfont)
    plt.axis('equal')
    plt.savefig(os.path.join(save_loc, save_name), dpi=300, bbox_inches="tight")

def summary_table(data, save_name, figsize=(8,6)):
    f, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.set_frame_on(False)
    table = pd.plotting.table(ax, data, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.8,1.2)
    plt.savefig(os.path.join(save_loc, save_name), dpi=300)

def overall_bar_chart(data, save_name):
    rcParams['font.size'] = 10
    f, ax = plt.subplots(figsize=(6,12))
    data = {i:j for i,j in data.items() if j != 0}
    plt.barh(list(data.keys()),list(data.values()),color=colors[0])
    plt.savefig(os.path.join(save_loc,save_name))
    rcParams['font.size'] = 14

def overall_plot(data, save_name):
    f, ax = plt.subplots(figsize=(8,6))
    data = {i:j for i,j in data.items() if j != 0}
    plt.scatter(list(data.keys()),list(data.values()), color=colors[0])
    plt.savefig(os.path.join(save_loc,save_name))

# set up trackers for each statistic
# patient information
sex_df = pd.DataFrame(
    data=np.zeros((len(repo_list)+1,3)),
    columns=['Female', 'Male', 'Not Specified'],
    index=repo_list+['Overall']
 )
COVID_df = pd.DataFrame(
    data=np.zeros((len(repo_list)+1,3)),
    columns=['Positive', 'Negative', 'Not Specified'],
    index=repo_list+['Overall']
)
race_df = pd.DataFrame(
    data = np.zeros((len(repo_list)+1, 1)),
    columns=['Not Specified'],
    index=repo_list+["Overall"]
)
ethnicity_df = pd.DataFrame(
    data = np.zeros((len(repo_list)+1, 1)),
    columns=['Not Specified'],
    index=repo_list+["Overall"]
)
age_df = pd.DataFrame(
  data = np.zeros((len(repo_list)+1, 1)),
    columns=['Not Specified'],
    index=repo_list+["Overall"]  
)
# image information
modality_df = pd.DataFrame(
    data=np.zeros((len(repo_list)+1,3)),
    columns=['CR', 'DX', 'Not Specified'],
    index=repo_list+['Overall']
)
body_part_df = pd.DataFrame(
    data = np.zeros((len(repo_list)+1, 1)),
    columns=['Not Specified'],
    index=repo_list+["Overall"]
)
# summarize
for repo in repo_list:
    # check repository
    if repo == 'RICORD-1c':
        df = pd.read_json(RICORD_1c_json_path, orient='table')
    elif repo == 'open_AI':
        df = pd.read_json(open_AI_json_path, orient='table')
    elif repo == 'COVID_19_AR':
        df = pd.read_json(COVID_19_AR_json_path, orient='table')
    else:
        print(f'ERROR unknown repository {repo}.')
        break
    # summarize each statistic
    sex_df = summarize_sex(sex_df, df, repo)
    COVID_df = summarize_COVID(COVID_df, df, repo)
    num_patients_with_CXR, num_CXR, modality_df = summarize_modality(modality_df, df, repo)
    race_df = summarize_race(race_df, df, repo)
    ethnicity_df = summarize_ethnicity(ethnicity_df, df, repo)
    age_df = summarize_age(age_df, df, repo)
    body_part_df = summarize_body_part(body_part_df, df, repo)


# find totals
sex_df.loc['Overall'] = sex_df.sum()
COVID_df.loc['Overall'] = COVID_df.sum()
modality_df.loc['Overall'] = modality_df.sum()
race_df.loc['Overall'] = race_df.sum()
ethnicity_df.loc['Overall'] = ethnicity_df.sum()
age_df.loc['Overall'] = age_df.sum()
body_part_df.loc['Overall'] = body_part_df.sum()
# organize
age_df = age_df[sorted(age_df.columns)]


 
if len(repo_list) == 1:
    pie_chart(sex_df.loc['Overall'], f"{repo}_sex_summary_chart.png")
    pie_chart(COVID_df.loc['Overall'], f"{repo}_COVID_summary_chart.png")
    pie_chart(modality_df.loc['Overall'], f"{repo}_modality_summary_chart.png", True, str(num_patients_with_CXR) + ' num. of patients with = ' + str(num_CXR) + ' CXR images')
    pie_chart(race_df.loc['Overall'],f"{repo}_race_summary_chart.png")
    pie_chart(ethnicity_df.loc['Overall'],f"{repo}_ethnicity_summary_chart.png")
    overall_bar_chart(age_df.loc['Overall'], f'{repo}_age_bar_chart.png')
    # overall_plot(age_df.loc['Overall'], f'{repo}_age_plot.png')
    # pie_chart(age_df.loc['Overall'],f'{repo}_age_summary_chart.png')
    pie_chart(body_part_df.loc['Overall'],f'{repo}_body_part_summary_chart.png')
else:
    pie_chart(sex_df.loc['Overall'], "sex_summary_chart.png")
    pie_chart(COVID_df.loc['Overall'],"COVID_summary_chart.png")
    pie_chart(modality_df.loc['Overall'],"modality_summary_chart.png")
    pie_chart(race_df.loc['Overall'],"race_summary_chart.png")
    pie_chart(ethnicity_df.loc['Overall'],"ethnicity_summary_chart.png")
    overall_bar_chart(age_df.loc['Overall'],'age_bar_chart.png')
    # overall_plot(age_df.loc['Overall'],"age_plot.png")
    # pie_chart(age_df.loc['Overall'],"age_summary_chart.png")
    pie_chart(body_part_df.loc['Overall'], "body_part_summary_chart.png")
    summary_table(age_df, "age_summary_table.png", figsize=(20,2))
    summary_table(sex_df, "sex_summary_table.png")

    


        



