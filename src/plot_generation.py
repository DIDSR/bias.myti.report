from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from datetime import date, datetime
from math import ceil
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.plot_formatting import *
 
def figure_plotting(
    data, x_col, s_col, hue_col, study_type, ylim=(0,1), y_label=None, x_label=None, style_col=None, 
    style_dict={}, color_dict={}, mean_col='Mean', lower_CI_col="lower_CI", upper_CI_col="upper_CI", plot_section=[]):  
    """
    Function to generate subplots with input plot sections and parameters.

    Arguments
    =========
    data
        dataframe that contains the data for plotting
    x_col
        name for column that contains x-axis ticks
    s_col
        name for column that contains sub-sections in the figure
    hue_col
        name for column that contains subgroups mapped with different colors during plotting
    ylim
        set the y-limits for all axes
    y_label
        set the y label name
    x_label
        set the x label name
    style_col
        name for column that determine line styles by positive-associated subgroup
    style_dict
        dictionary that determines plotting style
    color_dict
        dictionary that determins plotting colors
    mean_col
        name for column that contains metric mean value
    lower_CI_col
        name for column that contains lower bound of confidence interval
    upper_CI_col
        name for column that contains upper bound of confidence interval
    plot_section
        list that has all the sub-sections for plotting (including legends section)
    
    """
    fig = plt.figure(figsize = (text_width, text_width*FIGURE_RATIO))
    # # plot for baseline type study
    if study_type == 'None':
        gs = fig.add_gridspec(1,1)
        ax = fig.add_subplot(gs[0,0])
        ax.set_ylim(ylim)
        data = data[data[s_col] == plot_section[0]].sort_values(x_col).copy()
        ax.set_title(plot_section[0])
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        if style_col is not None:
          gb = [hue_col, style_col]
        else:
          gb = [hue_col]
        for gp, df in data.groupby(gb):
          hue = gp[0]
          if style_col:
            style = gp[-1]
          else:
            style = 'Default'          
          ax.fill_between(df[x_col], df[lower_CI_col], df[upper_CI_col], color=color_dict[hue], alpha=CI_ALPHA)
          ax.plot(df[x_col], df[mean_col], c=color_dict[hue], ls=style_dict[style])
        ax.set_xticks(data[x_col].unique().tolist())
        # set the "0" to "B" (for baseline)
        labels = ax.get_xticks().tolist()
        labels[0] = "B"
        ax.set_xticklabels(labels)
        
        name_map = SUBGROUP_NAME_MAPPING
        for h in data[hue_col].unique().tolist():
            if h not in name_map:
                name_map[h] = h
        hue_lines = [Patch(facecolor=color_dict[h], label=name_map[h]) for h in data[hue_col].unique().tolist()]
        hue_legend = ax.legend(handles=hue_lines,  title=hue_col, loc='upper left')
        if style_col:
          style_lines = [Line2D([0], [0], ls=style_dict[s], color='k', label=name_map[s]) for s in data[style_col].unique().tolist()]
          style_legend = ax.legend(handles=style_lines, title="Positive-Associated", loc='upper right')
          fig.add_artist(hue_legend)
    
    else:
        # # calculate the number of rows needed
        col_num = NUM_SUB_COL
        row_num = ceil(len(plot_section) /col_num)
        # # create figure with sub-sections        
        gs = fig.add_gridspec(row_num, col_num)
        data = data.sort_values(x_col)
        axes = []
        for r in range(row_num):
          for c in range(col_num):
            if (r * col_num + c + 1) <= len(plot_section):
                axes.append(fig.add_subplot(gs[r,c]))
                axes[-1].set_ylim(ylim)
                axes[-1].set_xticks(data[x_col].unique().tolist())
                if x_label:
                  axes[-1].set_xlabel(x_label)
                if c == 0:
                  axes[-1].set_ylabel(y_label)
                else:
                  axes[-1].set_yticklabels([])
        # # generate plots in each sub-sections
        for i, m in enumerate(plot_section):
          ax = axes[i]
          if study_type == "Study Finite Sample Size Effect":
              ax.set_title(f"{m} training set size")
          else:
              ax.set_title(m)
          # # plot legends in an individual sub-section
          if m == "legends":
            ax.axis("off")
            ax.set_title(None)
            ax.set_ylim(0,1)
            ax.set_xlim(0,1)
            name_map = SUBGROUP_NAME_MAPPING
            for h in data[hue_col].unique().tolist():
                if h not in name_map:
                    name_map[h] = h
            hue_lines = [Patch(facecolor=color_dict[h], label=name_map[h]) for h in data[hue_col].unique().tolist()]
            hue_legend = ax.legend(handles=hue_lines, bbox_to_anchor=(0.5,0.5), loc='lower center', title=hue_col)
            if style_col:
              style_lines = [Line2D([0], [0], ls=style_dict[s], color='k', label=name_map[s]) for s in data[style_col].unique().tolist()]
              style_legend = ax.legend(handles=style_lines, title="Positive-Associated", bbox_to_anchor=(0.5,0.5), loc='upper center')
              fig.add_artist(hue_legend)
          # # generate plots from the input data
          else:
            temp_data = data[data[s_col] == m].copy()
            if len(temp_data) == 0: # blank plot if no available data
              ax.text((ax.get_xlim()[1] - ax.get_xlim()[0])/2 + ax.get_xlim()[0], (ax.get_ylim()[1] - ax.get_ylim()[0])/2 + ax.get_ylim()[0], "N/A", va='center', ha='center', bbox=dict(facecolor='white', edgecolor='white'), fontsize=16)
              continue
            if style_col is not None:
              gb = [hue_col, style_col]
            else:
              gb = [hue_col]
            for gp, df in temp_data.groupby(gb):
              hue = gp[0]
              if style_col:
                style = gp[-1]
              else:
                style = 'Default'
              # plot line and confidence interval
              ax.fill_between(df[x_col], df[lower_CI_col], df[upper_CI_col], color=color_dict[hue], alpha=CI_ALPHA)
              ax.plot(df[x_col], df[mean_col], c=color_dict[hue], ls=style_dict[style])
            ax.set_xticks(data[x_col].unique().tolist())
            # set the "0" to "B" (for baseline)
            labels = ax.get_xticks().tolist()
            labels[0] = "B"
            ax.set_xticklabels(labels)    
    # # save the figure
    gs.tight_layout(fig, w_pad=0.1, h_pad=0.75)
    fig.savefig(os.path.join('../example/', f"figure_{y_label}.png"), bbox_inches='tight')
    plt.close("all")    

    

def calculate_CI(df, mean_col='Mean', std_col='Std', confidence_level=0.95, sample_size=25):
    """
    Function to calculate confidence interval according to standard deviation, confidence level and sample size.
    
    Arguments
    =========
    df
        dataframe that contains the data to calculate CI
    mean_col
        name for column with mean value
    std_col
        name for column with standard deviation
    confidence_level
        confidence level, currently support 0.9, 0.95 and 0.99
    sample size
        number of samples for each experiments in the data

    Returns
    =======
    pandas.DataFrame
        dataframe that adds computed lower and upper confidence interval bound
    """
    z_stats = {0.90:1.64, 0.95:1.96, 0.99:2.57}
    z = z_stats[confidence_level]
    df['lower_CI'] = df[mean_col] - z*(df[std_col] / (sample_size**(0.5)) )
    df['upper_CI'] = df[mean_col] + z*(df[std_col] / (sample_size**(0.5)) )
    return df

def bias_report_generation(variables, csv_path, exp_type, study_type):
    """
    Function to load inputs from the user, generate report figures and descriptions.

    Arguments
    =========
    variables
        dictionary that contains user specified column names for every variables
    csv_path
        path for the csv file which contains the data
    exp_type
        string to indicate the bias amplification type
    study_type
        string to indicate the study type

    Returns
    =======
    m_list : obj:`list`
        list contains plotted metrics
    info_list : obj:`list`
        list contains report description text corresponding to each metric
    """
    data = pd.read_csv(csv_path)
    # # get user input positive-associated group and metric value column for plotting         
    plot_kwargs = dict(style_col=variables.get('Positive-associated Subgroup'), hue_col=variables.get('Subgroup'), mean_col=variables.get('Metric Mean Value'), color_dict=COLORS, style_dict=STYLES)  
    # # get study type to establish sub-sections
    if study_type == 'Compare Bias Mitigation Methods':
        s_col = variables.get('Mitigation Method')
        section_name = data[s_col].unique().tolist()
        if 'No Mitigation' in section_name:
            section_name.insert(0, section_name.pop(section_name.index('No Mitigation')))
        section_name.append("legends")
    elif study_type == 'Study Finite Sample Size Effect':
        s_col = variables.get('Training Data Size')
        section_name = sorted(data[s_col].unique().tolist())
        section_name.append("legends")
    elif study_type == 'None':
        data['Section Name'] = 'Bias amplification'
        s_col = 'Section Name'
        section_name = data[s_col].unique().tolist()
    else:
        raise NotImplementedError()
    # # add a separate section to plot legends    
    plot_kwargs['study_type'] = study_type
    plot_kwargs['s_col'] = s_col
    plot_kwargs['plot_section'] = section_name
    # # get bias amplification type
    if exp_type == 'Quantitative Misrepresentation':        
        x_col = variables.get('Training Prevalence Difference')
        degree_name = "Training Prevalence Difference (%)"
    elif exp_type == 'Inductive Transfer Learning':
        x_col = variables.get('Frozen Layers')
        degree_name = "Frozen Layers"
    else:
        raise NotImplementedError()
    plot_kwargs["x_col"] = x_col
    plot_kwargs["x_label"] = degree_name
    # # calculate confidence interval 
    data = calculate_CI(data, mean_col=variables.get('Metric Mean Value'), std_col=variables.get('Metric Standard Deviation'))
    # # get list of metrics for plotting
    m_col = variables.get('Metric Name')
    m_list = list(data[m_col].unique())
    info_list = []
    # # plot each metric in a loop
    for i, m in enumerate(m_list):
        temp_data = data[(data[m_col] == m)].copy() # data for current metric
        # # plotting
        figure_plotting(data=temp_data.copy(), y_label=m, **plot_kwargs)        
        # # generate corresponding report description
        # according to study type
        if study_type == "None":
            info = f"The figure presents the subgroup {m} when model bias has been amplified by {exp_type.lower()}."
        elif study_type == "Compare Bias Mitigation Methods":
            info = f"The figure compares the subgroup {m} between different bias mitigation methods when bias is amplified by {exp_type.lower()}." + \
            "The first subplot presents the amplified bias (without mitigation), and the rest subplots show results from different implemented mitigation methods."
        else:            
            info = f"The figure compares the subgroup {m} between models with different training set sizes when bias is amplified by {exp_type.lower()}." + \
            "Subplots in the figure present results with different sample sizes used for model training."
        # according to amplification type
        if exp_type == "Quantitative Misrepresentation":
            info = info + "For these experiments, the positive-associated subgroup refers to the subgroup with the higher disease prevalence in the training set." + \
            "The x-axis indicates the subgroup disease prevelance difference in the training set, while B indicates the baseline model."
        else:
            info = info + "For these experiments, the positive-associated subgroup refers to the subgroup associated with the same model output during extra transfer learning step." + \
            "The x-axis indicates the number of layers being frozen during the final model fine-tune step, while B indicates the baseline model."                    
        info_list.append(info)
    return m_list, info_list
    