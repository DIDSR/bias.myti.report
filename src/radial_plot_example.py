from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import math
import os

from plot_formatting import *

RCOLOR = "#2fa6a8"
THETACOLOR= "#5e269e"
ANNOTCOLOR = 'black'


# =================================

def to_cartesian(r, theta, theta_type = 'degrees'):
    if theta_type == 'degrees':
      theta = math.radians(theta)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y
       
def create_example_plot(save_loc, approach='direct', figsize=(8,6), fmt='png', annotation_degrees=45, n_sections=5, center_hole_portion=0.2, rlim=(0,1), pad_degrees=5, pad_width=True):
    fig, ax = plt.subplots(figsize=figsize)
    
    theta_text = "Degree of\nbias amplification"
    
    ax.axis("off")
    ax.set_aspect('equal')
    
    si, section_coords, label_anchor = make_zoomed_out(ax, ylim=[0,0.4], xlim=[0,0.4], annotation_degrees=annotation_degrees, center_hole = center_hole_portion, n_sections=n_sections)
    
    box_coords = make_zoomed_in(ax, ylim=[0.4,1], xlim=[0.4,1], center_hole = center_hole_portion, section_inc = si, thetaaxis=theta_text)
    
    # draw the lines from the section to the zoomed in portion
    for i in [0,1]:
      ax.plot( [section_coords[i][0], box_coords[i][0]], [section_coords[i][1], box_coords[i][1]], color=ANNOTCOLOR)
    
    # make the section label box
    section_label_box(ax, label_anchor)
    
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    if pad_width:
      factor = figsize[0]/figsize[1] - 1
      ax.set_xlim(-factor/2, 1+factor/2)
    #plt.savefig(os.path.join(save_loc, f"radial_example_{approach}.{fmt}"), bbox_inches='tight', pad_inches=1)
    plt.savefig(os.path.join(save_loc, f"radial_example.{fmt}"), bbox_inches='tight', pad_inches=0.1)
    plt.close('all')
    return
    
def section_label_box(ax, anchor, color='#48a142', offset=(0.3,0)):
    """ Makes a label box for the different sections. """
    text = "Each section shows\na different bias\nmitigation method"
    ax.annotate(text, xy=anchor, xytext=[anchor[i] + o for i, o in enumerate(offset)], arrowprops=dict(arrowstyle="-|>", edgecolor=color), bbox=dict(edgecolor=color, fill=False), verticalalignment='center', horizontalalignment='center', color=color)
    return   
    
def make_zoomed_out(ax, xlim, ylim, annotation_degrees, center_hole, n_sections):
    """ Draws the zoomed out portion and provides the section increment for use in drawing the zoomed in portion and the coordinates needed for the zoom lines """
    center = [sum(xlim)/2, sum(ylim)/2]
    outer_rad = sum(xlim)/2
    inner_rad = outer_rad*center_hole
    ax.add_collection( PatchCollection([
      patches.Arc(center, width=outer_rad*2, height=outer_rad*2, theta2=0, theta1=annotation_degrees, angle=90, edgecolor='k'), # outer arc
      patches.Arc(center, width=inner_rad*2, height=inner_rad*2, theta2=0, theta1=annotation_degrees, angle=90, edgecolor='k'), # inner arc
    ], match_original=True, clip_on=False))
    section_inc = (360 - annotation_degrees)/n_sections
    section_borders = np.linspace(0, (360-annotation_degrees), n_sections+1)
    for sb in section_borders:
      x1, y1 = to_cartesian(outer_rad, sb+90+annotation_degrees)
      x2, y2 = to_cartesian(inner_rad, sb+90+annotation_degrees)
      ax.plot([x1+center[0],x2+center[0]], [y1+center[1], y2+center[1]], lw=2, color='k')
    # get the first sections's coordinates
    x1, y1 = to_cartesian(outer_rad, 0+90)
    x2, y2 = to_cartesian(outer_rad, 0-section_inc+90)
    
    # Get the coordinate of the center of the second section
    x3, y3 = to_cartesian(outer_rad, 0+90-(section_inc*1.5))
    
    return section_inc, ( (x1+center[0],y1+center[1]), (x2+center[0],y2+center[1]) ), (x3+center[0], y3+center[1])
    
def make_zoomed_in(ax, xlim, ylim, section_inc, center_hole, n_rticks=5, thetaticks=[0,50,80,100], pad_degree=5,raxis="Performance\nmeasurement", thetaaxis="Degree of bias amplification"):
    """ Draw the zoomed in seciton and it's bounding box; return the two corners needed to draw the zoom lines """
    # draw box
    ax.add_collection(PatchCollection([patches.Rectangle((xlim[0], ylim[0]), width=xlim[1]-xlim[0], height=ylim[1]-ylim[0], fill=False, edgecolor=ANNOTCOLOR)], match_original=True, clip_on=False))
    center = [xlim[0]+(0.15*sum(xlim)/2), ylim[0]] # slight adjustment to make room for text
    outer_rad = 0.65*(xlim[1]-xlim[0])
    inner_rad = outer_rad*center_hole
    # draw arcs
    
    p = []
    for r in np.linspace(inner_rad, outer_rad, n_rticks+2):
      if r == inner_rad or r == outer_rad:
        alpha=1
      else:
        alpha=0.2
      p.append( patches.Arc(center, width=r*2, height=r*2, theta1=0, theta2=section_inc, angle=90-section_inc, edgecolor='k', alpha=alpha) )
    ax.add_collection( PatchCollection(p, match_original=True))
    
    # Draw lines
    for sb in [0, section_inc]:
      x1, y1 = to_cartesian(outer_rad, 90-sb)
      x2, y2 = to_cartesian(inner_rad, 90-sb)
      ax.plot([x1+center[0],x2+center[0]], [y1+center[1], y2+center[1]], lw=2, color='k')
      
    inc = (section_inc - (pad_degree*2)) / max(thetaticks)
    for t in thetaticks:
      theta = pad_degree + t*inc
      x1, y1 = to_cartesian(outer_rad, 90-theta)
      x2, y2 = to_cartesian(inner_rad, 90-theta)
      ax.plot([x1+center[0],x2+center[0]], [y1+center[1], y2+center[1]], lw=2, color='k', alpha=0.2)
      # add ticklabel
      x3, y3 = to_cartesian(outer_rad*1.01, 90-theta)
      ax.text(x3+center[0], y3+center[1], str(t), color='k', alpha=0.2)
    
    style = "Simple, tail_width=0.005, head_width=.015, head_length=.01"
    # raxis label and arrow
    x1, y1 = to_cartesian(inner_rad, 90)
    x2, y2 = to_cartesian(outer_rad, 90)
    
    ax.add_collection(PatchCollection([patches.FancyArrowPatch( (x1+center[0]-0.015, y1+center[1]), (x2+center[0]-0.015, y2+center[1]), arrowstyle=style, color=RCOLOR)], match_original=True))
    ax.text(x1+center[0]-0.02, y1+center[1]+(outer_rad-inner_rad)/2, raxis, color=RCOLOR, rotation=90, ha='center', rotation_mode='anchor', va='bottom')
    
    # theta axis label and arrow
    x1, y1 = to_cartesian(outer_rad*1.21, 90)
    x2, y2 = to_cartesian(outer_rad*1.21, 90-section_inc)
    
    ax.add_collection(PatchCollection([patches.FancyArrowPatch( (x1+center[0], y1+center[1]), (x2+center[0], y2+center[1]), arrowstyle=style, color=THETACOLOR, connectionstyle='arc3,rad=-0.33')], match_original=True))
    
    x3, y3 = to_cartesian(outer_rad*1.4, 90-(section_inc/2))
    
    ax.text(x3+center[0], y3+center[1], thetaaxis, color=THETACOLOR, ha='center', va='bottom')
    
    return ((xlim[0], ylim[1]), (xlim[1], ylim[0]))
    
if __name__ == "__main__":
    save_loc = "/home/alexis.burgon/temp/bias_mit__example_figues/"
    for app in ['direct']:
        create_example_plot(save_loc, approach=app)