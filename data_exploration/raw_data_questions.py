
#%%
import pandas as pd
import os

from PersonAI.anchor import data_dir
# data path
BASE_PATH = os.path.join(data_dir(),'Structure Test Person AI','Structure Test Person AI')

DATA_FILE_PATH = os.path.join(BASE_PATH,"raw")
df_raw = pd.read_excel(os.path.join(DATA_FILE_PATH,"id001","id001_availability.xlsx"))#,index_col = 0)#Erhebungsbogen vollständig(1-93).xlsx'),index_col=0)  
# df = df.rename(columns={df.columns[0]:"id","Startzeit":"timestamp"})
# %%
df_raw = pd.read_excel(os.path.join(DATA_FILE_PATH,"id001","id001_availability.xlsx"))#,index_col = 0)#Erhebungsbogen vollständig(1-93).xlsx'),index_col=0)  
#%%
DATA_PATH = data_dir()

df_cold = pd.read_csv(os.path.join(DATA_PATH,"final_main_dataset_no_missing_ALLENC.csv"))#,index_col = 0)#Erhebungsbogen vollständig(1-93).xlsx'),index_col=0)  

# %%
participant_count = df_cold.user_id.value_counts()
participant_count = participant_count.rename_axis('unique_values').reset_index(name='counts')
participant_count["participant_weight"] = sum(participant_count["counts"])/participant_count["counts"]
# %%
