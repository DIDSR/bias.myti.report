from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import data_management as dm
import pandas as pd
import numpy as np
import math
import os

# Note: these colors are not the exact same hex values as used in the other plots
COLORS = {"F":"#E3289C", "M":"#1E7AD0", "Overall":"#57BB67"}  
HIGHLIGHT_COLOR = "#FDCA40"
BIAS_DEG_COLOR = "#FE7F2D"

def radial_plot(mitigation_methods:list, data, degree_col="Training Prevalence Difference", ymax=0.75, ymin=0.5,metric='AUROC', figsize=(8,6)):
  
  data = data.sort_values(degree_col)
  
  fig, ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=figsize)
  
  ax.set_theta_direction(-1)
  ax.set_theta_zero_location("N")
  
  annot_degrees = 45
  
  xmax = math.radians(360-annot_degrees)
  
  ax.set_xlim(0, xmax)
  ax.set_ylim(ymin, ymax)
  
  
  # for polar plots theta = x axis, radial = r axis
  
  inc = xmax / (len(mitigation_methods)*2)
  adj = 0
  
  new_theta_ticks = np.arange(0 + adj, xmax + adj, inc)
  
  label_ticks = new_theta_ticks[1::2]
  theta_ticks = new_theta_ticks[0::2]
  
  for i in range(len(new_theta_ticks)):
    if new_theta_ticks[i] > 2*math.pi:
      new_theta_ticks[i] -= 2*math.pi
  
  ax.set_xticks(theta_ticks, labels=["" for t in theta_ticks])
  ax.set_ylim(ymin,ymax)
  
   
  
  for t in theta_ticks:
    ax.annotate("", [t, ymin], [t, ymax*1.05], arrowprops=dict(arrowstyle="-", lw=1))
  
  pad = 5 # number of degrees to not plot between for each section
  pad *= 2*math.pi / 360
  inc *= 2
  inc /= (data[degree_col].max() - data[degree_col].min())
  inc -= (pad/data[degree_col].max() - data[degree_col].min())
  for i, m in enumerate(mitigation_methods):
    # add section label
    rotate = -1*math.degrees(label_ticks[i])
    if abs(rotate) > 120 and abs(rotate) < 240:
      rotate -= 180
    ax.text(label_ticks[i], ymax*1.05, m, rotation=rotate, rotation_mode="anchor", ha='center', fontweight='bold')
    # get relevent region
    if theta_ticks[i] == theta_ticks[-1]:
      region = [theta_ticks[i]+pad/2, theta_ticks[0]-pad/2]
    else:
      region = [theta_ticks[i]+pad/2, theta_ticks[i+1]-pad/2]
      
    if m == "No Mitigation":
      # highlight the no mitigation portion
      n_points = 100
      start = region[0]-pad/2
      stop = region[1]+pad/2
      temp_inc = (stop-start)/n_points
      x_values = np.arange(start, stop+temp_inc, inc)
      ax.fill_between(x_values, [ymax for x in x_values], [ymin for x in x_values], color=HIGHLIGHT_COLOR, zorder=0, alpha=0.5)
    # get each degree
    degrees = [deg for deg in data[degree_col].unique()]
    degree_theta = []
    
    tick = [ymax*0.99, ymax*1.01]    
    
    
    for deg in degrees:
      t = region[0] + inc*(deg)
      if t > 2*math.pi:
        t -= 2*math.pi
      degree_theta.append(t)
      ax.annotate("", [t, tick[0]], [t, tick[1]], arrowprops=dict(arrowstyle="-"))
      ax.text(t, ymax*1.015, str(deg), rotation=-1* math.degrees(t), rotation_mode="anchor", ha='center')
      
    for subg in data['subgroup'].unique():
      color = COLORS[subg]
      mean_values = []
      upper_CI = []
      lower_CI = []
      temp_data = data[(data['mitigation'] == m) & (data['subgroup'] == subg)].copy()
      if len(temp_data) == 0:
        continue
      
      for deg in degrees:
        assert len(temp_data[temp_data[degree_col] == deg]) == 1
      
        mean_values.append(temp_data[temp_data[degree_col] == deg]['Mean'].values[0])
        upper_CI.append(temp_data[temp_data[degree_col] == deg]['upper_CI'].values[0])
        lower_CI.append(temp_data[temp_data[degree_col] == deg]['lower_CI'].values[0])
      
      # plot CI
      ax.fill_between(degree_theta, lower_CI, upper_CI, color=color, alpha=0.3)
      # plot mean values
      ax.plot(degree_theta, mean_values, c=color)
      
  # make axis labels
  ax.text(math.radians(-(annot_degrees*0.5)), (ymax+ymin)/2, metric, rotation=90, rotation_mode="anchor", ha='center', fontweight='bold')
  ax.annotate("", [0, ymax], [math.radians(-annot_degrees),ymax],arrowprops=dict(arrowstyle="->", lw=1, connectionstyle="arc3, rad=-0.2",color=BIAS_DEG_COLOR))
  ax.text(math.radians(-annot_degrees/2), ymax*1.02, "Training Prevalence\nDifference (%)", ha='center', rotation=annot_degrees/2, rotation_mode="anchor",color=BIAS_DEG_COLOR)
  
  yticklabels = [f"{y:.2f}" for y in ax.get_yticks()]
  yticklabels[0] = ""
  yticklabels[-1] = ""
  ax.set_yticks(ax.get_yticks(), labels=yticklabels)
  
  # make legend
  custom_lines = [
    Line2D([0], [0], color=COLORS["F"], lw=1,),
    Line2D([0], [0], color=COLORS["M"], lw=1,),
    Line2D([0], [0], color=COLORS["Overall"], lw=1,),
  ]
  custom_labels = ["Female", "Male", "Overall"]
  fig.legend(custom_lines, custom_labels, loc='upper right',bbox_to_anchor=(0.9,0.9))
  
    
  return fig, ax 
  
def calculate_CI(df, mean_col='mean', std_col='std', confidence_level=0.95, sample_size=25):
  z_stats = {0.90:1.64, 0.95:1.96, 0.99:2.57}
  z = z_stats[confidence_level]
  df['lower_CI'] = df['Mean'] - z*(df['Std'] / (sample_size**(0.5)) )
  df['upper_CI'] = df['Mean'] + z*(df['Std'] / (sample_size**(0.5)) )
  return df
  
def import_data(main_dir="/gpfs_projects/alexis.burgon/OUT/2022_CXR/Bias_manipulation_manuscript/results"):
  data = {}
  for dir in os.listdir(main_dir):
    if os.path.isdir(os.path.join(main_dir, dir)):
      amp = dir.split("-")[0]
      exp = dir.split("-")[1]
      if amp not in data.keys():
        data[amp] = {}
        
      df = dm.load_data(os.path.join(main_dir, dir), amp)
      details = {"metric":['AUROC', 'Sensitivity', 'Prevalence']}
      details['subgroup'] = ["Overall"] + [x for x in df['Positive-associated'].unique() if x != 'N/A']
      df = dm.unpack_details(df, details)
      df = dm.clean_data(df, mitigation={'baseline':'No Mitigation', 'reject_object_classification':'Reject Option Classification', 'image_cropping':'Image Cropping', 'reweighing':"Reweighing", 'calibrated_equalized_odds':"Calibrated Equalized Odds"})
      df['subgroup'] = df['subgroup'].fillna("Overall")
      data[amp][exp] = df.copy()
  return data
  
if __name__ == "__main__":
  print("Starting...")
  save_dir = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/Bias_manipulation_manuscript/test_images/"
  metrics = ['AUROC']
  data = import_data()
  
  data = data['direct']['sex'].copy()
  
  data = data[data['task'] == 'COVID'].copy()
  
  data = calculate_CI(data)
  for metric in metrics:
    for PA in data["Positive-associated"].unique():
      if PA == "N/A":
        continue
      temp_data = data[(data["Positive-associated"].isin([PA, "N/A"])) & (data.metric == metric)].copy()
      
      print(temp_data["Positive-associated"].unique(), temp_data["Training Prevalence Difference"].unique())
      
      fig, ax = radial_plot(mitigation_methods = ["No Mitigation","Image Cropping", "Reweighing", "Calibrated Equalized Odds", "Reject Option Classification",], data=temp_data, metric=metric)
      for fmt in ['svg','png']:
        plt.savefig(os.path.join("/gpfs_projects/alexis.burgon/OUT/2022_CXR/Bias_manipulation_manuscript/image_of_the_semester_plots/", f"radial_{PA}_positive-associated_{metric}.{fmt}"), bbox_inches='tight')
  print("Done")