from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import data_management as dm
import pandas as pd
import numpy as np
import textwrap
import math
import os

from plot_formatting import *

  
class RadialPlot():
    def __init__(self, section_names:list, data, x_col, degree_name=None, rmax=1, rmin=0, axes_kwargs={}, figsize=(8,6), title=None, r_label=None):
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=figsize)
        self.rlim = [rmin, rmax]
        if degree_name is None:
          self.degree_name = x_col
        else:
          self.degree_name = degree_name
        self.format_axes(**axes_kwargs)
        self.init_sections(section_names, data, x_col)
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
        self.ax.tick_params(axis="y", pad=0.5)
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
        
    def init_sections(self, section_names, data, x_col):
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
        #self.sections["No Mitigation"].highlight()
          
    def plot(self, section_col='mitigation', **plot_kwargs):
        for nm, section in self.sections.items():
            data = self.data[self.data[section_col] == nm].copy()
            section.plot(data, **plot_kwargs) 
        self.legend(**plot_kwargs)
        
    def legend(self, pad=0.0, name_map_dict=SUBGROUP_NAME_MAPPING, **kwargs):
        name_mapping = SUBGROUP_NAME_MAPPING
        for h in self.data[kwargs['hue_col']].unique().tolist():
          if h not in name_mapping:
            name_mapping[h] = h
        hue_lines = [Patch(facecolor=kwargs['color_dict'][h], label=name_mapping[h]) for h in self.data[kwargs['hue_col']].unique().tolist()]
        hue_legend = self.fig.legend(handles=hue_lines, loc='upper right', bbox_to_anchor=[1-pad,1-pad])
        if 'style_col' in kwargs:
          style_lines = [Line2D([0],[0], ls=kwargs['style_dict'][s], color='k', label=name_mapping[s]) for s in self.data[kwargs['style_col']].unique().tolist()]
          style_legend = self.fig.legend(handles=style_lines, loc='upper left',bbox_to_anchor=[0+pad,1-pad], title=kwargs['style_col'].title())
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
            
    def interpolate_points(self, x_values, *values, step_every=1):
        """ Interpolates such that there at least one point every step_every. 
        Note: assumes that the values are sorted according to the x_value order """       
        steps = np.arange(min(x_values), max(x_values), step=step_every)
        steps = list(set([*steps, *x_values])) # so that we don't remove any original values
        steps.sort()
        
        final_output = [steps]
        for val in values:
            original_points = dict(zip(x_values, val))
            output = []
            for x in steps:
                if x in x_values: # was one of the original values
                    idx = np.where(x_values == x)
                    output.append(val[idx][0])
                else: # linear interpolation
                    # find upper and lower bounds for interpolation
                    lower = max(x_values[np.where(x_values < x)])
                    upper = min(x_values[np.where(x_values > x)])
                    lower_idx = np.where(x_values == lower)
                    upper_idx = np.where(x_values == upper)
                    
                    # define the linear function
                    slope = ( val[upper_idx] - val[lower_idx] ) / ( upper - lower )
                    
                    output.append((slope*(x - lower) + val[lower_idx])[0])
            final_output.append(output) 
        return final_output   
        
    def plot(self, data, mean_col, x_col, hue_col,  color_dict, style_col=None, style_dict={}, CI=True, lower_CI_col="lower_CI", upper_CI_col="upper_CI", interpolate=False, step_every=1):
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
          mean_values = df[mean_col].values
          lower_CI_values = df[lower_CI_col].values
          upper_CI_values = df[upper_CI_col].values          
          
          if interpolate:
              x_values, mean_values, lower_CI_values, upper_CI_values = self.interpolate_points(x_values, mean_values, lower_CI_values, upper_CI_values, step_every=step_every)
          
          # convert from x to theta
          theta_values = [self.convert_point(x) for x in x_values]
          
          if CI:
            self.RP.ax.fill_between(theta_values, lower_CI_values, upper_CI_values, color=color_dict[hue], alpha=CI_ALPHA)
          
          self.RP.ax.plot(theta_values, mean_values, c=color_dict[hue], ls=style_dict[style]) 
          
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


def result_plotting(variables, csv_path, exp_type, study_type):
    data = pd.read_csv(csv_path)  
    
    plot_kwargs = dict(style_col=variables.get('Positive Associated Subgroup'), mean_col=variables.get('Metric Mean Value'), color_dict=COLORS, style_dict=STYLES, interpolate=True)  
    if study_type == 'Compare Bias Mitigation Methods':
        s_col = variables.get('Mitigation Method')
        section_name = data[s_col].unique()
    elif study_type == 'Study Finite Sample Size Effect':
        s_col = variables.get('Training Data Size')
        section_name = data[s_col].unique()

    if exp_type == 'Quantitative Misrepresentation':        
        x_col = variables.get('Training Prevalence Difference')
        degree_name = "Training Prevalence Difference (%)"
        plot_kwargs['step_every'] = 10
    elif exp_type == 'Inductive Transfer Learning':
        x_col = variables.get('Frozen Layers')
        degree_name = "Frozen Layers"
        plot_kwargs['step_every'] = 1
    else:
        raise NotImplementedError()
    plot_kwargs["x_col"] = x_col 
    data = calculate_CI(data, mean_col=variables.get('Metric Mean Value'), std_col=variables.get('Metric standard deviation'))

    m_col = variables.get('Metric Type')
    for i, m in enumerate(data[m_col].unique()):
        temp_data = data[(data[m_col] == m)].copy()

        if m == 'AUROC':
            kwargs = dict(rmax=0.9, rmin=0.5, r_label=m)
        elif m == 'Prevalence':
            kwargs = dict(rmax=1, rmin=0, r_label="Predicted\nPrevalence (%)")

        RP = RadialPlot(section_names=section_name, data=temp_data.copy(), x_col=x_col, axes_kwargs=dict(center_hole_portion=0.2), degree_name=degree_name, **kwargs)
        RP.plot(section_col=s_col, **plot_kwargs, hue_col=variables.get('Subgroup'))
        plt.savefig(os.path.join('../example/', f"example_{i}.png"), bbox_inches='tight')
        plt.close("all")
  