import os
import requests
from datetime import date
import json

main_dir = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823"

def get_version_info(main_dir=main_dir):
    fp = os.path.join(main_dir, "data_version_tracking.json")
    if os.path.exists(fp):
        confirm = input("data version tracking file already exists, overwrite? [y] yes [n] no\n")
        while confirm not in ['y','n', 'yes', 'no']:
            confirm = input("invalid entry. Overwrite? [y] yes [n] no\n")
        if confirm == 'n' or confirm == 'no':
            print("Not overwriting, returning")
            return
        print("Overwriting tracking file..")
    # get release version TODO: test once we have a release (worked on covidGR repo, but that's public!)
    response = requests.get("https://api.github.com/repos/ravisamala/summarize_CXR_data/releases/latest")
    if response.status_code == 404:
        # prerelease
        rel = "0.0.0"
    else:
        rel = response.json()['name']
    
    info_dict = {'release version':rel, 'date':date.today().strftime("%Y/%m/%d")}
    with open(fp, 'w') as f:
        json.dump(info_dict, f)

if __name__ == '__main__':
    get_version_info()