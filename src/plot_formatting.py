from matplotlib import rcParams

# Formatting details
text_width = 8 # in inches
COLORS = {"F":"#dd337c", "M":"#0fb5ae", "Overall":"#57c44f", "B":"#4046ca", "W":"#f68511", "COVID":"#57c44f", "Subgroup":"#7e84fa",} 
STYLES = {"Default":"-", "M":"--", "F":"-", "B":"-", "W":"--"} # positive-associated
ACCENT_COLOR = "#430c82" # used for bias deg indicator arrow and accompanying text
HIGHLIGHT_COLOR = "#FDCA40"
SUBGROUP_NAME_MAPPING = {"F":"Female", "M":"Male","B":"Black", "W":"White"} # for the legend(s)
CI_ALPHA = 0.3
AXIS_COLOR = "#6e6e6e"
NUM_SUB_COL = 3 # number of plots in a row in the figure
FIGURE_RATIO = 0.75

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
