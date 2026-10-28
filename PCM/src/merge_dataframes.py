import pandas
from src_anchor import src_dir

# merge ghazals bpm measurement age feature with the other features

def main():
    wbpm = pandas.read_csv(src_dir() + '../dataset/all_aircontrol_with_surveys_hr_with_new_bpm_new.csv')
    df = pandas.read_csv(src_dir() + '../dataset/final_main_dataset_no_missing_ALLENC_with_time.csv')
    df = df.set_index(['user_id','timestamp'])
    wbpm = wbpm.rename(columns={'survey_time': 'timestamp'})
    wbpm = wbpm.set_index(['user_id','timestamp'])
    new_cols = wbpm[['bpm', 'bpm_time_diff_log']]

    wer =df.merge(right=new_cols, how='outer', on=['user_id', 'timestamp']).reset_index()
    wer = wer.dropna(subset='hr_ubi')
    print(wer[wer['bpm_time_diff_log'].isna()])
    print(len(df))
    print(len(wer))
    wer.to_csv(src_dir() + '../dataset/final_main_dataset_no_missing_ALLENC_with_time_and_hr_age.csv')

if __name__ == '__main__':
    main()