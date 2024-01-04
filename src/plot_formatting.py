from matplotlib import rcParams

COLORS = {"F":"#dd337c", "M":"#0fb5ae", "Overall":"#72e06a", "B":"#4046ca", "W":"#f68511", "COVID":"#72e06a", "Subgroup":"#7e84fa",} 
STYLES = {"Default":"-", "M":"--", "F":"-", "B":"-", "W":"--"} # positive-associated
ACCENT_COLOR = "#430c82" # used for bias deg indicator arrow and accompanying text
HIGHLIGHT_COLOR = "#FDCA40"
SUBGROUP_NAME_MAPPING = {"F":"Female", "M":"Male","B":"Black", "W":"White"} # for the legend(s)
CI_ALPHA = 0.3

# Matplotlib Rc parameters (importing this file will apply them)
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelsize'] = 16
rcParams['axes.titlesize'] = 18
rcParams['font.size'] = 12
rcParams['font.weight'] = 'bold'
rcParams['grid.alpha'] = 0.5
