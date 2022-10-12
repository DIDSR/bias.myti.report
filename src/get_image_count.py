import os
import json
import pandas as pd

main_dir = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/"
repos = ['COVID_19_NY_SBU', 'COVID_19_AR', 'MIDRC_RICORD_1C', 'open_AI', 'open_RI']

def get_count():
    out_df = pd.DataFrame(columns=['repo','patient_count', 'image_count'])
    for repo in repos:
        # print(repo)
        input_file = os.path.join(main_dir, f"summary_table__{repo}.json")
        in_df = pd.read_json(input_file, orient='table')
        # in_df = in_df[in_df.num_images > 0]
        in_df = in_df[in_df['images'].map(lambda d: len(d)>0)]
        patient_count = len(in_df)
        image_count = 0
        for ii, each_patient in in_df.iterrows():
            if type(each_patient['images']) == list:
                image_count += len(each_patient['images'])
                if repo == 'open_AI':
                    if len(each_patient['images_info']) < len(each_patient['images']):
                        print("images", len(each_patient['images']))
                        print('image_info', len(each_patient['images_info']))
                    len_list = [len(img_name) for img_name in each_patient['images']]
                    if sum(len_list)/len(len_list) < 200:
                        print(len_list)

                    # if len(each_patient['images']) > 20:
                    #     print(each_patient['images'][0])
            else:
                # print("Not a list?")
                return
        out_df.loc[len(out_df)] = [repo, patient_count, image_count]
    out_df = out_df.set_index('repo')
    out_df.loc['total'] = out_df.sum()
    # out_df.to_csv(os.path.join(main_dir, 'total_image_counts.csv'))
    print(out_df)

def check_jpeg():
    for repo in repos:
        print(repo)
        jpeg_dir = os.path.join(main_dir, f"{repo}_jpegs")
        print(len(os.listdir(jpeg_dir)))

if __name__ == "__main__":
    get_count()
    # check_jpeg()
    