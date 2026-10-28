#%%
import pandas as pd
import os
import numpy as np
import re

from anchor import src_dir
#%%

# data path
BASE_PATH = os.path.join(src_dir())

df = pd.read_excel(os.path.join(BASE_PATH,"data","personai",'Onboarding survey_all participants_cleaned.xlsx'))#,index_col = 0)#Erhebungsbogen vollständig(1-93).xlsx'),index_col=0)  
df = df.rename(columns={df.columns[0]:"id","Startzeit":"timestamp"})

df = df [~(df ["id"].isin(["Valerie Test","Valerie Test Handy", "test", "test Katja echt"])) & (~(df.loc[:,"id"].isna()))]
df = df.iloc[1:,:]
df.loc[:,'id'] = df['id'].apply(lambda x: "id" + re.findall(r'[0-9]+',x)[0].zfill(3))
df = df.reset_index(drop = True)

def sr(score,base):
    #score reverse
    return base-score

# %%

results = df.iloc[:,0].to_frame()
# %% Kältescore 
# reverse score 5
temperatur = df.iloc[:,34:40]

results["Hitze_Tolleranz"] = temperatur.apply(lambda x: sum([sr(x[0],5),sr(x[1],5),sr(x[2],5)]), axis=1)
results["Kälte_Tolleranz"] = temperatur.apply(lambda x: sum([sr(x[3],5),sr(x[4],5),sr(x[5],5)]), axis=1)

#%%

results["SWLS"] = df.iloc[:,40:45].apply(lambda x: sum(x), axis=1)

# %% MAIA: 
# Reverse: 5-score

MAIA = pd.concat([df.iloc[:,0],df.iloc[:,45:82]], axis=1) #MAIA2 
results["MAIA_Bemerken"] = MAIA.apply(lambda x: np.mean(x[1:5]), axis=1)
results["MAIA_Nicht-Ablenken"] = MAIA.apply(lambda x: np.mean(sr(x[5:11],5)), axis=1)
results["MAIA_ Sich-Keine-Sorgen-Machen"] = MAIA.apply(lambda x: np.mean([sr(x[11],5),sr(x[12],5),x[13],x[14],sr(x[15],5)]), axis=1)
results["MAIA_Aufmerksamkeitsregulation"] = MAIA.apply(lambda x: np.mean(x[16:23]), axis=1)
results["MAIA_Emotionales Gewahrsein"] = MAIA.apply(lambda x: np.mean(x[23:28]), axis=1)
results["MAIA_Selbstregulation"] = MAIA.apply(lambda x: np.mean(x[28:32]), axis=1)
results["MAIA_Auf-den-Leib-Hören"] = MAIA.apply(lambda x: np.mean(x[32:34]), axis=1)
results["MAIA_Vertrauen"] = MAIA.apply(lambda x: np.mean(x[35:38]), axis=1)


#%%
BFI2S = pd.concat([df.iloc[:,0],df.iloc[:,82:112]],axis = 1) #BFI-2-S

results["BFI2S_Extraversion"] = BFI2S.apply(lambda x: sum([sr(x[1],6),x[6],x[11],x[16],sr(x[21],6),sr(x[26],6)]), axis=1)
results["BFI2S_Geselligkeit"] = BFI2S.apply(lambda x: sum([sr(x[1],6),x[16]]), axis=1)
results["BFI2S_Durchsetzungsfähigkeit"] = BFI2S.apply(lambda x: sum([x[6],sr(x[21],6)]), axis=1)
results["BFI2S_Aktivität"] = BFI2S.apply(lambda x: sum([x[11],sr(x[26],6)]), axis=1)

results["BFI2S_Verträglichkeit"] = BFI2S.apply(lambda x: sum([x[2],sr(x[7],6),x[12],sr(x[17],6),x[22],sr(x[27],6)]), axis=1)
results["BFI2S_Mitgefühl"] = BFI2S.apply(lambda x: sum([x[2],sr(x[17],6)]), axis=1)
results["BFI2S_Höflichkeit"] = BFI2S.apply(lambda x: sum([sr(x[7],6),x[22]]), axis=1)
results["BFI2S_Vertrauen"] = BFI2S.apply(lambda x: sum([x[12],sr(x[27],6)]), axis=1)

results["BFI2S_Gewissenhaftigkeit"] = BFI2S.apply(lambda x: sum([sr(x[3],6),sr(x[8],6),x[13],x[18],x[23],sr(x[28],6)]), axis=1)
results["BFI2S_Ordnungsliebe"] = BFI2S.apply(lambda x: sum([sr(x[3],6),x[18]]), axis=1)
results["BFI2S_Fleiß"] = BFI2S.apply(lambda x: sum([sr(x[8],6),x[23]]), axis=1)
results["BFI2S_Verlässlichkeit"] = BFI2S.apply(lambda x: sum([x[13],sr(x[28],6)]), axis=1)

results["BFI2S_Negative Emotionalität"] = BFI2S.apply(lambda x: sum([x[4],x[9],sr(x[14],6),sr(x[19],6),sr(x[24],6),x[29]]), axis=1)
results["BFI2S_Emotionalität"] = BFI2S.apply(lambda x: sum([sr(x[4],6),sr(x[9],6),x[14],x[19],x[24],sr(x[29],6)]), axis=1)

results["BFI2S_Ängstlichkeit"] = BFI2S.apply(lambda x: sum([x[4],sr(x[19],6)]), axis=1)
results["BFI2S_Niedergeschlagenheit"] = BFI2S.apply(lambda x: sum([x[9],sr(x[24],6)]), axis=1)
results["BFI2S_Unbeständigkeit der Gefühle"] = BFI2S.apply(lambda x: sum([sr(x[14],6),x[29]]), axis=1)

results["BFI2S_Offenheit"] = BFI2S.apply(lambda x: sum([x[5],sr(x[10],6),x[15],sr(x[20],6),x[25],sr(x[30],6)]), axis=1)
results["BFI2S_Ästhetisches Empfinden"] = BFI2S.apply(lambda x: sum([x[5],sr(x[20],6)]), axis=1)
results["BFI2S_Intellektuelle Neugierde"] = BFI2S.apply(lambda x: sum([sr(x[10],6),x[25]]), axis=1)
results["BFI2S_ Kreativer Einfallsreichtum"] = BFI2S.apply(lambda x: sum([x[15],sr(x[30],6)]), axis=1)

#%%
BFI2XS = pd.concat([df.iloc[:,0],df.iloc[:,82:97]],axis = 1)#.columns #BFI-2-XS

results["BFI2XS_Extraversion"] = BFI2XS.apply(lambda x: sum([sr(x[1],6),x[6],x[11]]), axis=1)
results["BFI2XS_Verträglichkeit"] = BFI2XS.apply(lambda x: sum([x[2],sr(x[7],6),x[12]]), axis=1)
results["BFI2XS_Gewissenhaftigkeit"] = BFI2XS.apply(lambda x: sum([sr(x[3],6),sr(x[8],6),x[13]]), axis=1)
results["BFI2XS_Negative Emotionalität"] = BFI2XS.apply(lambda x: sum([x[4],x[9],sr(x[14],6)]), axis=1)
results["BFI2XS_Emotionalität"] = BFI2XS.apply(lambda x: sum([sr(x[4],6),sr(x[9],6),x[14]]), axis=1)
results["BFI2XS_Offenheit"] = BFI2XS.apply(lambda x: sum([x[5],sr(x[10],6),x[15]]), axis=1)

#%%
BSI = pd.concat([df.iloc[:,0],df.iloc[:,112:]],axis = 1) # BSI

results["BSI_Somatization"] = BSI.apply(lambda x: np.nanmean([x[2],x[7],x[23],x[29],x[30],x[33],x[37]]), axis=1)
results["BSI_Obsession-Compulsion"] = BSI.apply(lambda x: np.nanmean([x[5],x[15],x[26],x[27],x[32],x[36]]), axis=1)
results["BSI_Interpersonal Sensitivity"] = BSI.apply(lambda x: np.nanmean([x[20],x[21],x[22],x[42]]), axis=1)
results["BSI_Depression"] = BSI.apply(lambda x: np.nanmean([x[9],x[16],x[17],x[18],x[35],x[50]]), axis=1)
results["BSI_Anxiety"] = BSI.apply(lambda x: np.nanmean([x[1],x[12],x[19],x[38],x[45],x[49]]), axis=1)
results["BSI_Hostility"] = BSI.apply(lambda x: np.nanmean([x[6],x[13],x[40],x[41],x[46]]), axis=1)
results["BSI_Phobic Anxiety"] = BSI.apply(lambda x: np.nanmean([x[8],x[28],x[31],x[43],x[47]]), axis=1)
results["BSI_Paranoid Ideation"] = BSI.apply(lambda x: np.nanmean([x[4],x[10],x[24],x[48],x[51]]), axis=1)
results["BSI_Psychoticism"] = BSI.apply(lambda x: np.nanmean([x[3],x[14],x[34],x[44],x[52]]), axis=1)
# %%

results["BSI_GSI"] = df.iloc[:,112:].apply(lambda row: np.nanmean(row),axis = 1)
results["BSI_PST"] = df.iloc[:,112:].apply(lambda row:  np.count_nonzero(row[~np.isnan(row)]),axis = 1)
results["BSI_PDSI_temp"] = df.iloc[:,112:].apply(lambda row: sum(row[~np.isnan(row)]),axis = 1)
results["BSI_PDSI"] = results["BSI_PDSI_temp"]/results["BSI_PST"]
results = results.drop("BSI_PDSI_temp",axis='columns' )
# %%

results_with_info = pd.concat([df.iloc[:,:34],results.iloc[:,1:]],axis = 1)
# %%
results_with_info.to_csv(os.path.join(BASE_PATH,"data","personai",'Scores.csv'), index=False)

# %%
results_with_info.to_csv(os.path.join(BASE_PATH,"data","personai",'Scores.csv'), index=False)
# %%
