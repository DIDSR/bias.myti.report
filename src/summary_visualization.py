import seaborn
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

# formatting
colors = seaborn.color_palette("Paired")
rcParams['font.family'] = 'monospace'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 14
hfont = {'fontname':'monospace', 'fontweight':'bold', 'fontsize':18}
rcParams['figure.figsize'] = (8,6)
f, ax = plt.subplots(figsize=(8,6))

# read in dataframe
RICORD_1c_json_path = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/summary_table_RICORD_1c.json"
df = pd.read_json(RICORD_1c_json_path, orient='table')
# set up save location
save_loc = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/RICORD_1c_sex_visualization.png"
# set up summary lists
sex_counts = [0,0]
sex_labels = ['Female', 'Male']
for row in df.itertuples():
    if row[4][0]['sex'] == 'F':
        sex_counts[0] += 1
    elif row[4][0]['sex'] == 'M':
        sex_counts[1] += 1
    else:
        print(f"ERROR, got {row[4][0]['sex']} instead of F or M")
# print(sex_counts)
# create pie chart
plt.pie(sex_counts, labels=sex_labels, colors=colors, autopct='%.0f%%')
plt.savefig(save_loc, dpi=300)

