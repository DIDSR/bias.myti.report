from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import data_management as dm
import pandas as pd
import numpy as np
import textwrap
import math
import os

COLORS = {"F":"#dd337c", "M":"#0fb5ae", "Overall":"#72e06a", "B":"#4046ca", "W":"#f68511", "COVID":"#72e06a", "Subgroup":"#7e84fa",} 
STYLES = {"Default":"-", "M":"--", "F":"-", "B":"-", "W":"--"} # positive-associated
ACCENT_COLOR = "#430c82" # used for bias deg indicator arrow and accompanying text
HIGHLIGHT_COLOR = "#FDCA40"
SUBGROUP_NAME_MAPPING = {"F":"Female", "M":"Male"} # for the legend(s)

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

  
class RadialPlot():
    def __init__(self, section_names:list, data, x_col, degree_name=None, rmax=1, rmin=0, axes_kwargs={}, figsize=(8,6), title=None, r_label=None):
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=figsize)
        self.rlim = [rmin, rmax]
        if degree_name is None:
          self.degree_name = x_col
        else:
          self.degree_name = degree_name
        self.format_axes(**axes_kwargs)
        self.init_sections(section_names)
        self.data = data.sort_values(x_col)
        if title is not None:
          self.ax.set_title(title, fontweight='bold')
        if r_label is not None:
          self.ax.text(math.radians(360-17), (rmax+rmin)/2, r_label, rotation=90, ha='right', va='center', fontweight='bold')
        
    def format_axes(self,center_hole_portion=0, annotation_degrees=45, hide_first_r=True, hide_last_r=True, n_r_ticks=6):
        self.ax.set_ylim(self.rlim)
        r_ticks = np.linspace(self.rlim[0], self.rlim[1], num=n_r_ticks)
        r_tick_labels = [f"{r:.2f}" for r in r_ticks]
        if hide_first_r:
          r_tick_labels[0] = ""
        if hide_last_r:
          r_tick_labels[-1] = ""
        self.ax.set_yticks(r_ticks, r_tick_labels)
        self.ax.set_xlim(0, math.radians(360-annotation_degrees))
        self.ax.set_xticks([])
        self.ax.set_theta_direction(-1) # clockwise
        self.ax.set_theta_zero_location("N")
        
        if center_hole_portion <= 0:
            r_origin = 0
        else:
            r_origin = (self.rlim[1] - self.rlim[0]) / (0.5/center_hole_portion) - self.rlim[0]
            
        self.ax.set_rorigin(-r_origin)
        self.total_r = self.rlim[1] - self.rlim[0]
        self.section_label_r = self.rlim[1] + self.total_r*0.1
        
        # draw the degree arrow (and accompanying text)
        p1 = math.radians(360 - annotation_degrees)
        p2 = math.radians(0)
           
        self.ax.annotate("", [p2, self.rlim[1]], [p1, self.rlim[1]],arrowprops=dict(arrowstyle="->", lw=1, connectionstyle="arc3, rad=-0.2",color=ACCENT_COLOR))
        self.ax.text(math.radians(360 - (annotation_degrees/2)), self.rlim[1] + self.total_r*0.05, "\n".join(textwrap.wrap(self.degree_name, 20)),
                     color=ACCENT_COLOR, ha="center", va="bottom", rotation=annotation_degrees/2, rotation_mode='anchor')
        
    def init_sections(self, section_names):
        _, xmax = self.ax.get_xlim()
        rmin, rmax = self.ax.get_ylim()
        section_size = xmax / len(section_names)
        section_dividers = np.arange(0, xmax, section_size)
        self.sections = {}
        for i, nm in enumerate(section_names):
            self.sections[nm] = RadialPlotSection(self, rp_xmin=section_dividers[i], rp_xmax=section_dividers[i]+section_size, name=nm, x_values=data[x_col].unique())
        # Draw the section dividers
        for sd in [s for s in section_dividers] + [xmax]:
          self.ax.plot([sd,sd], [rmin, rmax+self.total_r*0.05], lw=2, color='k')[0].set_clip_on(False)
        self.sections["No Mitigation"].highlight()
          
    def plot(self, section_col='mitigation', **plot_kwargs):
        for nm, section in self.sections.items():
            data = self.data[self.data[section_col] == nm].copy()
            section.plot(data, **plot_kwargs) 
        self.legend(**plot_kwargs)
        
    def legend(self, pad=0.05, name_map_dict=SUBGROUP_NAME_MAPPING, **kwargs):
        name_mapping = SUBGROUP_NAME_MAPPING
        for h in self.data[kwargs['hue_col']].unique().tolist():
          if h not in name_mapping:
            name_mapping[h] = h
        hue_lines = [Patch(facecolor=kwargs['color_dict'][h], label=name_mapping[h]) for h in self.data[kwargs['hue_col']].unique().tolist()]
        hue_legend = self.fig.legend(handles=hue_lines, loc='lower left', bbox_to_anchor=[0+pad,0+pad])
        if 'style_col' in kwargs:
          style_lines = [Line2D([0],[0], ls=kwargs['style_dict'][s], color='k', label=name_mapping[s]) for s in self.data[kwargs['style_col']].unique().tolist()]
          style_legend = self.fig.legend(handles=style_lines, loc='lower right',bbox_to_anchor=[1-pad,0+pad], title=kwargs['style_col'].title())
          self.fig.add_artist(hue_legend)
        
class RadialPlotSection():
    def __init__(self, radial_plot, rp_xmin, rp_xmax, name, x_values, pad_degree=5, label_wrap=15, ha_adj=15):
        self.RP = radial_plot
        # constants
        self.ticklabel_pad = 0.01
        self.ticklabel_pos = self.RP.total_r*self.ticklabel_pad + self.RP.rlim[1]
        self.tick_alpha = 0.2
        self.rp_xlim = [rp_xmin, rp_xmax]
        
        # label the section
        center = (rp_xmax + rp_xmin)/2
        ha, va = self.get_alignment(center)
        
        if len(name) > label_wrap:
            lines = textwrap.wrap(name, label_wrap)
            label = "\n".join(lines)
        else:
            label = name
        
        self.RP.ax.text(center, self.RP.section_label_r, label, ha=ha, va=va, fontweight='bold')
        # Set up unit conversion
        rp_xmax -= math.radians(pad_degree)
        rp_xmin += math.radians(pad_degree)
        rp_xrange = rp_xmax - rp_xmin
        xrange = x_values.max() - x_values.min()
        self.conversion_factor = rp_xrange / xrange
        self.b = rp_xmin
        
        # mark theta ticks
        self.xticks(points=x_values)
        
    def convert_point(self, point):
        return self.conversion_factor*point + self.b
        
    def get_alignment(self, rad:float, ha_adj=15):
        d = math.degrees(rad)
        # # Determine the text alignment
        if d > ha_adj and d < 180-ha_adj:
            ha = 'left'
        elif d > 180+ha_adj and d < 360-ha_adj:
            ha = 'right'
        else:
            ha = 'center'
        if d > 90 and d < 270:
            va = 'top'
        else:
            va = 'bottom'
        
        return ha, va
        
    def xticks(self, points:list, zero_as_baseline=True):
        for p in points:
            theta = self.convert_point(p)
            self.RP.ax.plot([theta, theta], self.RP.rlim, color='k', alpha=self.tick_alpha)[0].set_clip_on(False)
            ha, va = self.get_alignment(theta)
            if zero_as_baseline and p == 0:
              text = "B"
            else:
              text = str(p)
            self.RP.ax.text(theta, self.ticklabel_pos, text, color='k', alpha=self.tick_alpha, ha=ha, va=va)
            
    def plot(self, data, mean_col, x_col, hue_col,  color_dict, style_col=None, style_dict={}, CI=True, lower_CI_col="lower_CI", upper_CI_col="upper_CI"): # TODO: CI
        assert mean_col in data.columns
        assert x_col in data.columns
        assert hue_col in data.columns
        if CI:
          assert lower_CI_col in data.columns
          assert upper_CI_col in data.columns
        
        if style_col is not None:
          gb = [hue_col, style_col]
        else:
          gb = [hue_col]
        for gp, df in data.groupby(gb):
          hue = gp[0]
          if style_col is not None:
            style = gp[1]
          else:
            style='Defualt'
          x_values = df[x_col].values
          # convert from x to theta
          theta_values = [self.convert_point(x) for x in x_values]
          
          if CI:
            self.RP.ax.fill_between(theta_values, df[lower_CI_col].values, df[upper_CI_col].values, color=color_dict[hue], alpha=0.3)
          
          self.RP.ax.plot(theta_values, df[mean_col].values, c=color_dict[hue], ls=style_dict[style]) 
          
    def highlight(self, n_points=100):
      inc = (self.rp_xlim[1] - self.rp_xlim[0]) / n_points
      x_values = np.arange(self.rp_xlim[0], self.rp_xlim[1]+inc, inc)
      self.RP.ax.fill_between(x_values, [self.RP.rlim[1] for x in x_values], [self.RP.rlim[0] for x in x_values], color=HIGHLIGHT_COLOR, zorder=0, alpha=0.5)
       
def calculate_CI(df, mean_col='mean', std_col='std', confidence_level=0.95, sample_size=25):
    z_stats = {0.90:1.64, 0.95:1.96, 0.99:2.57}
    z = z_stats[confidence_level]
    df['lower_CI'] = df['Mean'] - z*(df['Std'] / (sample_size**(0.5)) )
    df['upper_CI'] = df['Mean'] + z*(df['Std'] / (sample_size**(0.5)) )
    return df
  
if __name__ == "__main__":
    save_loc = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/Bias_manipulation_manuscript/radial_plots/"
    mitigation_methods = ["No Mitigation","Image Cropping", "Reweighing", "Calibrated Equalized Odds", "Reject Option Classification",]
    plot_kwargs = dict(style_col='Positive-associated', mean_col='Mean', color_dict=COLORS, style_dict=STYLES)
    all_data = import_data()
    for app in ['direct', 'indirect']:
      if app == 'direct':
        x_col = "Training Prevalence Difference"
        degree_name = "Training Prevalence Difference (%)"
      elif app == 'indirect':
        x_col = "Frozen Layers"
        degree_name = "Frozen Layers"
      else:
        raise NotImplementedError()
      plot_kwargs["x_col"] = x_col
      for exp in ['race', 'sex']:
        data = all_data[app][exp].copy()
        
        if app == 'indirect':
          # need x_col tp be numbers
          data.loc[data[x_col] == 'Baseline', x_col] = "0"
          data[x_col] = data[x_col].apply(lambda x: int(x))
          # remove reweighing, can't be applied to indirect approach
          if "Reweighing" in mitigation_methods:
            mitigation_methods.remove("Reweighing")
        
        data_list = []
        for PA in data['Positive-associated'].unique():
          if PA == "N/A":
            continue
          temp_data = data[data['Positive-associated'].isin([PA, "N/A"])].copy()
          temp_data['Positive-associated'] = PA
          data_list.append(temp_data)
        data = pd.concat(data_list)
        
        
        data = calculate_CI(data)
        
        # separate plots
        for compare in ['subgroup', 'task']:
            for m in ['AUROC', "Prevalence"]:
                if compare == 'subgroup':
                  temp_data = data[(data['metric'] == m) & (data['task'] == 'COVID')].copy()
                elif compare == 'task':
                  temp_data = data[(data['metric'] == m) & (data['subgroup'] == 'Overall')].copy()
                
                if len(temp_data) == 0:
                  continue
                  
                if m == 'AUROC':
                  kwargs = dict(rmax=0.9, rmin=0.5, r_label=m)
                elif m == 'Prevalence':
                  kwargs = dict(rmax=1, rmin=0, r_label="Predicted\nPrevalence (%)")
                
                #print(f"{app}_{exp}_{compare}_{m}")
                RP = RadialPlot(section_names=mitigation_methods, data=temp_data.copy(), x_col=x_col, axes_kwargs=dict(center_hole_portion=0.2), degree_name=degree_name, **kwargs)
                RP.plot(**plot_kwargs, hue_col=compare)
                
                for fmt in ["png"]:
                  plt.savefig(os.path.join(save_loc, f"{app}_{exp}_{compare}_{m}.{fmt}"), bbox_inches='tight')
                plt.close("all")
            
