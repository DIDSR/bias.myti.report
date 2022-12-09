import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn

import util

NEG_INF = -1e9


class Predictor(object):
    """Predictor class for a single model."""
    def __init__(self, model, device, code_dir):

        self.model = model
        #self.model = torch.nn.DataParallel(model, device_ids=[2])
        self.device = device
        self.code_dir = code_dir

    def predict(self, loader, by_patient=False, return_embeddings=False):
        if by_patient and not loader.dataset.return_info_dict:
            raise Exception("Cannot predict by patient when return_info_dict is False")
        self.model.eval()
        probs = []
        gt = []
        all_embeddings = []
        if loader.dataset.return_info_dict:
            paths = []
        with tqdm(total=len(loader.dataset)) as progress_bar:
            for data in loader:
                with torch.no_grad():
                    if loader.dataset.study_level:
                        if loader.dataset.return_info_dict:
                            inputs, targets, info_dict, mask = data
                        else:
                            inputs, targets, mask = data
                            #inputs, targets = data
                        # Fuse batch size `b` and study length `s`
                        b, s, c, h, w = inputs.size()
                        inputs = inputs.view(-1, c, h, w)

                        # Predict
                        logits, embeddings = self.model(inputs.to(self.device))
                        all_embeddings.append(embeddings.detach().cpu().numpy())
                        logits = logits.view(b, s, -1)

                        # Mask padding to negative infinity
                        ignore_where = (mask == 0).unsqueeze(-1)
                        ignore_where = ignore_where.repeat(1, 1,
                                                           logits.size(-1))
                        ignore_where = ignore_where.to(self.device)
                        logits = torch.where(ignore_where,
                                             torch.full_like(logits, NEG_INF),
                                             logits)
                        batch_logits, _ = torch.max(logits, 1)

                    else:
                        if loader.dataset.return_info_dict:
                            inputs, targets, info_dict = data
                        else:
                            inputs, targets = data

                        # batch_logits, unknown_tensor = self.model(inputs.to(self.device))
                        batch_logits, batch_embeddings = self.model(inputs.to(self.device))
                        x = F.relu(batch_embeddings, inplace=False)
                        pool = nn.AdaptiveAvgPool2d(1)
                        x = pool(x).view(x.size(0), -1)
                        if len(all_embeddings) == 0:
                            all_embeddings = x.detach().cpu().numpy()
                        else:
                            all_embeddings = np.vstack((all_embeddings, x.detach().cpu().numpy()))
                        #print(f'batch logits: {batch_logits}')

                    if self.model.module.model_uncertainty:
                        batch_probs =\
                            util.uncertain_logits_to_probs(batch_logits)
                    else:
                        
                        batch_probs = torch.sigmoid(batch_logits)
                        
                        # Decision boundary code uses softmax
                        #batch_probs = torch.nn.Softmax(dim=1)(batch_logits)
                        #print(f'batch probs: {batch_probs}')
                        

                probs.append(batch_probs.cpu())
                gt.append(targets)
                if loader.dataset.return_info_dict:
                    paths.extend(info_dict['paths'])
                progress_bar.update(targets.size(0))

        # print(len(all_embeddings))
        # print(all_embeddings.shape)
        # concat = np.concatenate(all_embeddings)
        # all_embeddings = concat.reshape(len(concat), -1)
        
        probs_concat = np.concatenate(probs)
        gt_concat = np.concatenate(gt)
        
        # with open('cx_res18.npy', 'wb') as f: # # betsy does not like this, have to give absolute path! This needs to be fixed properly!
        if not self.code_dir:
            with open('cx_res18.npy', 'wb') as f: # # betsy does not like this, have to give absolute path!
                np.save(f, all_embeddings)
                np.save(f, gt_concat)
        else:
            with open(f'{self.code_dir}/cx_res18.npy', 'wb') as f: # # betsy does not like this, have to give absolute path!
                np.save(f, all_embeddings)
                np.save(f, gt_concat)
       
        
        tasks = self.model.module.tasks # Tasks decided at self.model.module.tasks.
        probs_dict =  {task: probs_concat[:, i] for i, task in enumerate(tasks)}
        gt_dict = {task: gt_concat[:, i] for i, task in enumerate(tasks)}
        if loader.dataset.return_info_dict:
            probs_dict['Path'] = paths
            gt_dict['Path'] = paths
        probs_df = pd.DataFrame(probs_dict)
        gt_df = pd.DataFrame(gt_dict)
        # probs_df = pd.DataFrame({task: probs_concat[:, i]
        #                          for i, task in enumerate(tasks)})
        # gt_df = pd.DataFrame({task: gt_concat[:, i] # Check how gt_df looks like.
        #                       for i, task in enumerate(tasks)})

        self.model.train()
        if by_patient:
           print("Getting by-patient predictions and gts")
           probs_df = convert_to_by_patient(probs_df)
           gt_df = convert_to_by_patient(gt_df)
        if loader.dataset.return_info_dict and not return_embeddings:
            return probs_df, gt_df, paths
        elif loader.dataset.return_info_dict and return_embeddings:
            return probs_df, gt_df, paths, all_embeddings
        
        if return_embeddings:
            return probs_df, gt_df, all_embeddings

        return probs_df, gt_df

def path_to_pid(img_path):
    img_name = img_path.split("/")[-1]
    pid = "_".join(img_name.split("_")[:-1])
    return pid

def convert_to_by_patient(df):
    new_df = df.rename(columns={'Path':'patient_id'})
    out_df = pd.DataFrame()
    new_df['patient_id'] = new_df['patient_id'].apply(path_to_pid)
    for col in new_df.columns:
        if col == 'patient_id':
            continue
        out_df[col] = new_df.groupby(['patient_id'])[col].mean()
    # new_df = new_df.groupby(['patient_id']).mean()
    return out_df
