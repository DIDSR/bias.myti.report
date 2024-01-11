from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
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

def plot_wrapper(plot_function, width_portion=1, aspect=0.75, **kwargs):
    fig = plt.figure(figsize = (text_width*width_portion, text_width*width_portion*aspect))
    gs, gs_kwargs = FUNCTION_DICT[plot_function](fig, **kwargs)
    gs.tight_layout(fig, **gs_kwargs)
    return fig


def color_style_sampler(fig):
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    for i, (g, c) in enumerate(COLORS.items()):
        ax.fill_between([0,1], [i-0.1, i-0.1], [i+0.1, i+0.1], color=c, alpha=CI_ALPHA)
        ax.plot([0,1], [i-0.05, i-0.05], c=c)
        ax.plot([0,1], [i+0.05, i+0.05], c=c, ls = "--")
    return gs, {}
    
def mitigation_comparison(fig, data, x_col, hue_col, ylim=(0,1), y_label=None, x_label=None, style_col=None, style_dict={}, color_dict={}, mean_col='Mean', lower_CI_col="lower_CI", upper_CI_col="upper_CI", compare_title=None):
    gs = fig.add_gridspec(2,3)
    data = data.sort_values(x_col)
    axes = []
    for r in range(2):
      for c in range(3):
        axes.append(fig.add_subplot(gs[r,c]))
        axes[-1].set_ylim(ylim)
        axes[-1].set_xticks(data[x_col].unique().tolist())
        if x_label:
          axes[-1].set_xlabel(x_label)
        if c == 0:
          axes[-1].set_ylabel(y_label)
        else:
          axes[-1].set_yticklabels([])
    mitigation_plots = ["(i) No Mitigation", "(ii) Image Cropping", "legends", "(iii) Reweighing", "(iv) Calibrated\nEqualized Odds", "(v) Reject Option\nBased Classification"]
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
        temp_data = data[data['Mitigation'] == m.split(") ")[-1].replace("\nBased","").replace("\n"," ")].copy()
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
    
def no_mitigation_plot(fig, data, x_col, hue_col, ylim=(0,1), figsize=(8,6), y_label=None, x_label=None, style_col=None, style_dict={}, color_dict={}, mean_col='Mean', lower_CI_col="lower_CI", upper_CI_col="upper_CI", compare_title=None, title=None):
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    ax.set_ylim(ylim)
    data = data[data['Mitigation'] == "No Mitigation"].sort_values(x_col).copy()
    ax.set_title(title)
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
    hue_legend = ax.legend(handles=hue_lines,  title=compare_title, loc='upper left')
    if style_col:
      style_lines = [Line2D([0], [0], ls=style_dict[s], color='k', label=name_map[s]) for s in data[style_col].unique().tolist()]
      style_legend = ax.legend(handles=style_lines, title="Positive-Associated", loc='upper right')
      fig.add_artist(hue_legend)
    
    return gs, {}
    

def calculate_CI(df, mean_col='mean', std_col='std', confidence_level=0.95, sample_size=25):
    z_stats = {0.90:1.64, 0.95:1.96, 0.99:2.57}
    z = z_stats[confidence_level]
    df['lower_CI'] = df['Mean'] - z*(df['Std'] / (sample_size**(0.5)) )
    df['upper_CI'] = df['Mean'] + z*(df['Std'] / (sample_size**(0.5)) )
    return df

FUNCTION_DICT = {"sampler": color_style_sampler, "compare":mitigation_comparison, "baseline": no_mitigation_plot}

def result_plotting(variables, csv_path, exp_type, study_type):
    data = pd.read_csv(csv_path)  
    
    
    plot_kwargs = dict(style_col=variables.get('Positive-associated Subgroup'), mean_col=variables.get('Metric Mean Value'), color_dict=COLORS, style_dict=STYLES)  
    #if study_type == 'Compare Bias Mitigation Methods':
    #    s_col = variables.get('Mitigation Method')
    #    section_name = data[s_col].unique()
    #elif study_type == 'Study Finite Sample Size Effect':
    #    s_col = variables.get('Training Data Size')
    #    section_name = data[s_col].unique()

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
    info_list = []
    for i, m in enumerate(data[m_col].unique()):
        temp_data = data[(data[m_col] == m)].copy()
        if m == 'AUROC':
          kwargs = dict(ylim=(0.5,0.9), y_label=m)
        elif m == 'Prevalence':
          kwargs = dict(ylim=(0,1), y_label="Predicted\nPrevalence (%)")
        kwargs['compare_title'] = 'Population'
        fig = plot_wrapper("compare", data=temp_data.copy(), x_label=degree_name, hue_col='Subgroup', **kwargs, **plot_kwargs)
        fig.savefig(os.path.join('../example/', f"example_{i}.png"), bbox_inches='tight')
        plt.close("all")
        info = f"Comparison of {m} value for each subgroup across bias mitigation methods\nwhen bias has been amplified by {exp_type} with different degrees.\nFor these experiments, the positive-associated subgroup\nrefers to the subgroup with the higher disease prevalence in the training set."
        info_list.append(info)
    return info_list
    

def save_report(info, img_path, save_path):
    fig = plt.figure(figsize = (text_width, text_width))
    gs = fig.add_gridspec(4,1)
    ax1 = fig.add_subplot(gs[:-1,:])
    images = plt.imread(img_path)
    ax1.axis("off")
    ax1.imshow(images)
    ax2 = fig.add_subplot(gs[-1,:])
    ax2.axis("off")
    ax2.set_title(None)
    ax2.set_ylim(0,1)
    ax2.set_xlim(0,1)      
    ax2.text(0.02, 0.65, info)
    logo = plt.imread('UI_assets/fda_logo.jpg')
    fig.figimage(logo, 1700, 10)
    fig.savefig(save_path, bbox_inches='tight')

    







    