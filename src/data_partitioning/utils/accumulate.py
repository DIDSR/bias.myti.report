import pandas as pd

def accumulate(indiv_dfs, partition, steps):
    """ Accumulates the samples from previous steps """
    for s in range(1,steps):
        current = indiv_dfs[f"Step {s}"][partition].copy()
        prev = indiv_dfs[f"Step {s-1}"][partition].copy()
        indiv_dfs[f"Step {s}"][partition] = pd.concat([current, prev], axis=0)
    return indiv_dfs

def replace(indiv_dfs, partition, steps, RAND, id_col, by_subgroup=False):
    """ Replaces a portion of the previous step with the new step """
    for s in range(1,steps):
        current = indiv_dfs[f"Step {s}"][partition].copy()
        prev = indiv_dfs[f"Step {s-1}"][partition].copy()
        if len(current) > len(prev):
            raise Exception(f"Cannot use replace with step {s} being larger than step {s-1}")
        if not by_subgroup:
            remove = prev.sample(n=len(current), random_state=RAND)
        else:
            remove_dfs = []
            for sub in prev.subgroup.unique():
                remove_dfs.append(prev[prev.subgroup == sub].sample(n=len(current[current.subgroup == sub]), random_state=RAND))
            remove = pd.concat(remove_dfs, axis=0)
        indiv_dfs[f"Step {s}"][partition] = pd.concat([current, prev[~prev[id_col].isin(remove[id_col])]], axis=0)
    return indiv_dfs

        
    
            