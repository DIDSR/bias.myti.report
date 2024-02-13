from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from datetime import date
from math import ceil
import os

# Formatting details
text_width = 8 # in inches - doesn't have to be exact, just consistent
COLORS = {"F":"#dd337c", "M":"#0fb5ae", "Overall":"#57c44f", "B":"#4046ca", "W":"#f68511", "COVID":"#57c44f", "Subgroup":"#7e84fa",} 
STYLES = {"Default":"-", "M":"--", "F":"-", "B":"-", "W":"--"} # positive-associated
ACCENT_COLOR = "#430c82" # used for bias deg indicator arrow and accompanying text
HIGHLIGHT_COLOR = "#FDCA40"
SUBGROUP_NAME_MAPPING = {"F":"Female", "M":"Male","B":"Black", "W":"White"} # for the legend(s)
CI_ALPHA = 0.3
AXIS_COLOR = "#6e6e6e"

# Matplotlib Rc parameters (importing this file will apply them)
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelsize'] = 8
rcParams['axes.titlesize'] = 10
rcParams['font.size'] = 8
rcParams['font.weight'] = 'bold'
rcParams['grid.alpha'] = 0.5
rcParams['savefig.dpi'] = 300
rcParams['axes.grid'] = True
rcParams['xtick.color'] = AXIS_COLOR
rcParams['ytick.color'] = AXIS_COLOR
rcParams['axes.labelcolor'] = AXIS_COLOR
rcParams['axes.edgecolor'] = AXIS_COLOR
 
def mitigation_comparison(
    fig, data, x_col, s_col, hue_col, ylim=(0,1), y_label=None, x_label=None, style_col=None, style_dict={}, color_dict={}, 
    mean_col='Mean', lower_CI_col="lower_CI", upper_CI_col="upper_CI", plot_section=[], compare_title=None):
    row_num = ceil(len(plot_section) / 3)
    gs = fig.add_gridspec(row_num,3)
    data = data.sort_values(x_col)
    axes = []
    for r in range(row_num):
      for c in range(3):
        if (r * 3 + c + 1) <= len(plot_section):
            axes.append(fig.add_subplot(gs[r,c]))
            axes[-1].set_ylim(ylim)
            axes[-1].set_xticks(data[x_col].unique().tolist())
            if x_label:
              axes[-1].set_xlabel(x_label)
            if c == 0:
              axes[-1].set_ylabel(y_label)
            else:
              axes[-1].set_yticklabels([])
    mitigation_plots = plot_section
    for i, m in enumerate(mitigation_plots):
      ax = axes[i]
      ax.set_title(m)
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
        hue_legend = ax.legend(handles=hue_lines, bbox_to_anchor=(0.5,0.5), loc='lower center', title=compare_title)
        if style_col:
          style_lines = [Line2D([0], [0], ls=style_dict[s], color='k', label=name_map[s]) for s in data[style_col].unique().tolist()]
          style_legend = ax.legend(handles=style_lines, title="Positive-Associated", bbox_to_anchor=(0.5,0.5), loc='upper center')
          fig.add_artist(hue_legend)
      else:
        temp_data = data[data[s_col] == m].copy()
        if len(temp_data) == 0:
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
          
          ax.fill_between(df[x_col], df[lower_CI_col], df[upper_CI_col], color=color_dict[hue], alpha=CI_ALPHA)
          ax.plot(df[x_col], df[mean_col], c=color_dict[hue], ls=style_dict[style])
        ax.set_xticks(data[x_col].unique().tolist())
        # set the "0" to "B" (for baseline)
        labels = ax.get_xticks().tolist()
        labels[0] = "B"
        ax.set_xticklabels(labels)
    return gs, dict(w_pad=0.1, h_pad=0.75)
    

def calculate_CI(df, mean_col='Mean', std_col='Std', confidence_level=0.95, sample_size=25):
    z_stats = {0.90:1.64, 0.95:1.96, 0.99:2.57}
    z = z_stats[confidence_level]
    df['lower_CI'] = df[mean_col] - z*(df[std_col] / (sample_size**(0.5)) )
    df['upper_CI'] = df[mean_col] + z*(df[std_col] / (sample_size**(0.5)) )
    return df

def result_plotting(variables, csv_path, exp_type, study_type):
    data = pd.read_csv(csv_path)  
    
    
    plot_kwargs = dict(style_col=variables.get('Positive-associated Subgroup'), mean_col=variables.get('Metric Mean Value'), color_dict=COLORS, style_dict=STYLES)  
    if study_type == 'Compare Bias Mitigation Methods':
        s_col = variables.get('Mitigation Method')
        section_name = data[s_col].unique().tolist()
    elif study_type == 'Study Finite Sample Size Effect':
        s_col = variables.get('Training Data Size')
        section_name = data[s_col].unique().tolist()
        section_name.sort()
    section_name.insert(0, "legends")
    plot_kwargs['s_col'] = s_col
    plot_kwargs['plot_section'] = section_name
    if exp_type == 'Quantitative Misrepresentation':        
        x_col = variables.get('Training Prevalence Difference')
        degree_name = "Training Prevalence Difference (%)"
    elif exp_type == 'Inductive Transfer Learning':
        x_col = variables.get('Frozen Layers')
        degree_name = "Frozen Layers"
    else:
        raise NotImplementedError()
    plot_kwargs["x_col"] = x_col 
    data = calculate_CI(data, mean_col=variables.get('Metric Mean Value'), std_col=variables.get('Metric Standard Deviation'))

    m_col = variables.get('Metric Name')
    m_list = list(data[m_col].unique())
    info_list = []
    for i, m in enumerate(m_list):
        temp_data = data[(data[m_col] == m)].copy()
        kwargs = dict(y_label=m)
        kwargs['compare_title'] = 'Population'
        fig = plt.figure(figsize = (text_width, text_width*0.75))
        gs, gs_kwargs = mitigation_comparison(fig, data=temp_data.copy(), x_label=degree_name, hue_col='Subgroup', **kwargs, **plot_kwargs)
        gs.tight_layout(fig, **gs_kwargs)
        fig.savefig(os.path.join('../example/', f"example_{i}.png"), bbox_inches='tight')
        plt.close("all")
        info = f"The report shows the results of {study_type.lower()} experiment \nwhen bias is amplified by {exp_type.lower()}.\n"
        if study_type == "Compare Bias Mitigation Methods":
            info = info + f"Subplots in the above figure are comparing bias mitigation methods with baseline.\n"
        elif study_type == "Study Finite Sample Size Effect":
            info = info + f"Subplots in the figure are comparing different training dataset size.\n"
        info = info + f"The y-axis shows the {m} value of prediction results on test set.\n"
        if exp_type == "Quantitative Misrepresentation":
            info = info + "The x-axis indicates the subgroup disease prevelance difference in the training set.\n" + \
            "The positive-associated subgroup refers to the subgroup with the higher disease prevalence\n in the training set."
        elif exp_type == "Inductive Transfer Learning":
            info = info + "The x-axis indicates the number of layers being frozen during the final model fine-tune.\n" + \
            "The positive-associated subgroup refers to the subgroup associated with the same model output\n during extra transfer learning step."
        info_list.append(info)
    return m_list, info_list
    

def save_report(info, img_path, save_path):
    info = info + "\n\nReport generated by: myti.report v1.0" + f"\nDate: {date.today()}"
    fig = plt.figure(figsize = (text_width, text_width))
    gs = fig.add_gridspec(4,1)
    ax1 = fig.add_subplot(gs[:-1,:])
    images = plt.imread(img_path)
    ax1.axis("off")
    ax1.set_title('Bias Amplification/Mitigation Report')
    ax1.imshow(images)
    ax2 = fig.add_subplot(gs[-1,:])
    ax2.axis("off")
    ax2.set_title(None)
    ax2.set_ylim(0,1)
    ax2.set_xlim(0,1)
    ax2.text(0.02, 0.25, info)
    logo = plt.imread('UI_assets/fda_logo.jpg')
    fig.figimage(logo, 1700, 10)
    fig.savefig(save_path, bbox_inches='tight')

    







    