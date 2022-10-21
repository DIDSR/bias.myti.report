import os
import pandas as pd
import json
import torch


main_dirs = ["/gpfs_projects/ravi.samala/OUT/2022_CXR/model_runs/scenario_1_v3/"]

def find_files(fname, direc):
    out_fps = []
    for root, dirs, files in os.walk(direc):
        for name in files:
            if fname in name:
                out_fps.append(os.path.join(root, name))
    return out_fps

def summarize_models():
    # set up summary dataframe
    df = pd.DataFrame(columns=['fp','model','repo','step','epoch','iteration', 'AUROC'])
    for direc in main_dirs:
        # find tracking files
        tracking_fps = find_files("tracking.log", direc)
        for tracking_fp in tracking_fps:
            with open(tracking_fp, 'rb') as fp:
                tracking_info = json.load(fp)
            if len(tracking_info['Models']) == 0:
                continue
            # print(tracking_info)
            for model in tracking_info['Models']:
                if "Complete" not in tracking_info['Models'][model]["Training"]['Progress']:
                    continue
                if 'open_AI' in model:
                    repo = 'open_AI'
                
                elif "MIDRC_RICORD_1C" in model:
                    repo = "MIDRC_RICORD_1C"
                mdl_list = model.split("_")
                step_n = mdl_list[mdl_list.index('step')+1] 
                ckpt_path = os.path.join("/".join(tracking_fp.split("/")[:-1]),model,'best.pth.tar')
                ckpt_dict = torch.load(ckpt_path)
                # print(ckpt_dict['ckpt_info'])
                # print(ckpt_path)

                
            
                df.loc[len(df)] = ["/".join(tracking_fp.split("/")[-3:-1]),model, repo,step_n,  ckpt_dict['ckpt_info']['epoch'], ckpt_dict['ckpt_info']['iteration'],ckpt_dict['ckpt_info']['custom-AUROC']]
            
    df.to_csv("/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/model_summary.csv", index=None)

def get_last_iter(model_dir):
    model_iters = {}
    for fp in os.listdir(model_dir):    
        if not fp.endswith(".pth.tar"):
            continue
        elif 'best' in fp:
            continue
        iter_num = int(fp.replace("iter_","").replace(".pth.tar", ""))
        model_iters[iter_num] = fp
    # find the largest iter number
    x = max(model_iters.keys())
    iter_fp = os.path.join(model_dir,model_iters[x])
    return iter_fp

def alt_summarize_models(main_dir = "/gpfs_projects/ravi.samala/OUT/2022_CXR/model_runs/open_A1_scenario_1_v3/"):
    df = pd.DataFrame(columns=['rand', 'step', 'epoch', 'iteration'])
    for R in range(5):
        rand_dir = os.path.join(main_dir, f"RAND_{R}_OPTION_0_custom_7_steps")
        for step in range(7):
            print(f"RAND: {R}, step {step}")
            # ckpt_path = os.path.join(rand_dir, f"CheXpert_LRcustom3_Epcustom3__step_{step}",'best.pth.tar')
            ckpt_path = get_last_iter(os.path.join(rand_dir, f"CheXpert_LRcustom3_Epcustom3__step_{step}"))
            ckpt_dict = torch.load(ckpt_path)
            df.loc[len(df)] = [R, step, ckpt_dict['ckpt_info']['epoch'],ckpt_dict['ckpt_info']['iteration']]
    df.to_csv("/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v3/LI_model_summary.csv", index=None)


if __name__ == '__main__':
    # summarize_models()
    print("starting model summarization")
    alt_summarize_models()
