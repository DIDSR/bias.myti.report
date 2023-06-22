
# Repository summary files
summary_files_openHPC = {
    'open_A1':"/gpfs_projects/ravi.samala/OUT/2022_CXR/data_summarization/20221010/20221010_summary_table__open_A1.json",
    'open_R1':"/gpfs_projects/ravi.samala/OUT/2022_CXR/data_summarization/20221023/20221023_summary_table__open_R1.json",
    "MIDRC_RICORD_1C":"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__MIDRC_RICORD_1C.json",
    "COVID_19_NY_SBU":"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__COVID_19_NY_SBU.json",
    "COVID_19_AR":"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__COVID_19_AR.json",
    'BraTS':"/gpfs_projects/alexis.burgon/DATA/BraTS/20230315/BraTS21-17_Mapping.csv"
}

summary_files_betsy = {
    'open_A1':"/scratch/alexis.burgon/2022_CXR/data_summarization/20221010/summary_table__open_A1.json",
    'open_R1':"/scratch/alexis.burgon/2022_CXR/data_summarization/20221023/summary_table__open_R1.json",
    "MIDRC_RICORD_1C":"/scratch/alexis.burgon/2022_CXR/data_summarization/20220823/summary_table__MIDRC_RICORD_1C.json",
    "COVID_19_NY_SBU":"/scratch/alexis.burgon/2022_CXR/data_summarization/20220823/summary_table__COVID_19_NY_SBU.json",
    "COVID_19_AR":"/scratch/alexis.burgon/2022_CXR/data_summarization/20220823/summary_table__COVID_19_AR.json"
}

# Dicom to jpeg conversion mapping files
conversion_files_openHPC = {
    'open_A1':"/gpfs_projects/ravi.samala/OUT/2022_CXR/data_summarization/20221010/20221010_open_A1_jpegs/conversion_table.json",
    'open_R1':"/gpfs_projects/ravi.samala/OUT/2022_CXR/data_summarization/20221023/20221023_open_R1_jpegs/conversion_table.json",
    "MIDRC_RICORD_1C":"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/MIDRC_RICORD_1C_jpegs/conversion_table.json",
    "COVID_19_NY_SBU":"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/COVID_19_NY_SBU_jpegs/conversion_table.json",
    "COVID_19_AR":"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/COVID_19_AR_jpegs/conversion_table.json",
    'BraTS':None
}

conversion_files_betsy = {
    'open_A1':"/scratch/alexis.burgon/2022_CXR/data_summarization/20221010/open_A1_jpegs/conversion_table.json",
    'open_R1':None,
    "MIDRC_RICORD_1C":"/scratch/alexis.burgon/2022_CXR/data_summarization/20220823/MIDRC_RICORD_1C_jpegs/conversion_table.json",
    "COVID_19_NY_SBU":"/scratch/alexis.burgon/2022_CXR/data_summarization/20220823/COVID_19_NY_SBU_jpegs/conversion_table.json",
    "COVID_19_AR":"/scratch/alexis.burgon/2022_CXR/data_summarization/20220823/COVID_19_AR_jpegs/conversion_table.json",
    'BraTS':None
}

# Test date files (open-A1/R1 only)

test_date_files_openHPC = {
    'open_A1':"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/study_to_test_date/20230306_open_A1.csv",
    'open_R1':"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/study_to_test_date/20230419_open_R1.csv"
}

test_date_files_betsy = {
    'open_A1':"/scratch/yuhang.zhang/OUT.data_summarization/20230306_open_A1.csv"
}

portable_files_openHPC = {
    'open_A1':"/gpfs_projects/ravi.samala/DATA/MIDRC3/20221010_open_A1_all_Imaging_Studies.tsv",
    "open_R1":""
}

def get_test_date_file(repository, betsy=False):
    if betsy:
        raise test_date_files_betsy[repository] if repository in test_date_files_betsy else None
    else:
        return test_date_files_openHPC[repository] if repository in test_date_files_openHPC else None

def get_portable_file(repository, betsy=False):
    if betsy:
        raise NotImplementedError()
    else:
        return portable_files_openHPC[repository] if repository in portable_files_openHPC else None

def get_repository_files(repository, betsy=False):
    """ Fetch the summary and conversion files for specified repository """
    if betsy:
        if repository in summary_files_betsy:
            return summary_files_betsy[repository], conversion_files_betsy[repository]
        else:
            raise Exception(f"Could not find summary and conversion files for the {repository} repository, repository options are: {summary_files_betsy.keys()}")
    else:
        if repository in summary_files_openHPC:
            return summary_files_openHPC[repository], conversion_files_openHPC[repository]
        else:
            raise Exception(f"Could not find summary and conversion files for the {repository} repository, repository options are: {summary_files_betsy.keys()}")

# Available patient/image attributes
CXR_patient_info = ['sex','race','ethnicity','COVID_positive','age']
CXR_image_info = ['modality', 'body_part_examined', 'view_position', 'study_date', 'manufacturer', 'manufacturer_model_name', 'pixel_spacing', 'image_size']

