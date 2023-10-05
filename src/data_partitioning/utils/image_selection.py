from constants import *
import pandas as pd

def select_patient_images(df, args, test_date_csv=None):
    """ limits the number of images/patient according to args.img_per_patient """
    if args.img_per_patient == 0: # not limiting the number of patients
        return df
    if args.id_col == 'Path':
        raise Exception("Cannot limit number of images per patient using id column 'Path'")
    count_df = df.groupby(args.id_col)['Path'].nunique().reset_index()
    num_patients = df[args.id_col].nunique()
    num_images = df['Path'].nunique()
    if args.img_limit_type == 'min':
        df = df[df[args.id_col].isin(count_df[count_df['Path'] > args.img_per_patient-1][args.id_col])]
    else:
        df_to_limit = df[df[args.id_col].isin(count_df[count_df['Path'] > args.img_per_patient-1][args.id_col])]
        if args.img_limit_method == 'random':
            limited_df = df_to_limit.groupby(args.id_col)["Path"].sample(n=args.img_per_patient, random_state=args.RAND)
        elif args.img_limit_method == 'first':
            limited_df = df_to_limit.sort_values('study_date').groupby(args.id_col)["Path"].sample(n=args.img_per_patient, random_state=args.RAND)
        elif args.img_limit_method == 'last':
            limited_df = df_to_limit.sort_values('study_date', ascending=False).groupby(args.id_col)["Path"].sample(n=args.img_per_patient, random_state=args.RAND)
        elif args.img_limit_method == 'PCR':
            limited_df = select_images_PCR(df, test_date_csv, args)
        
        if args.img_limit_type == 'exact':
            df = limited_df
        elif args.img_limit_type == 'max':
            df = pd.concat([df[~df[args.id_col].isin(limited_df)], limited_df], axis=0)
    print(f"Limiting images/patient reduced dataset from {num_images} images ({num_patients} patients) to {df['Path'].nunique()} images ({df[args.id_col].nunique()} patients)")
    return df
        
def select_images_PCR(df, test_date_csv, args):
    test_dates = pd.read_csv(test_date_csv).set_index('jpeg')
    # positive cases
    pos_df = df[df["COVID_positive"] == 'Yes'].copy()
    pos_df['days_from_study_to_test'] = pos_df['Path'].map(test_dates['days_from_study_to_positive_test'])
    # negative cases
    neg_df = df[df["COVID_positive"] == 'No'].copy()
    neg_df['days_from_study_to_test'] = neg_df['Path'].map(test_dates['days_from_study_to_negative_test'])
    df = pd.concat([pos_df, neg_df], axis=0)
    if not args.allow_null_PCR: # remove patients without relevent test
        df = df[~df['days_from_study_to_test'].isnull()]
        print("Removing samples without test date information.")
    else: # set samples without test dates to arbitrarily far in the future to deprioritize them
        df['days_from_study_to_test'] = df['days_from_study_to_test'].fillna(1000000000)
    df['days_from_study_to_test'] = df['days_from_study_to_test'].abs() # considering absolute number of days
    df = df.sort_values('days_from_study_to_test')
    df = df.groupby(args.id_col).apply(pd.DataFrame.head, n=args.img_per_patient).reset_index(drop=True)
    return df
