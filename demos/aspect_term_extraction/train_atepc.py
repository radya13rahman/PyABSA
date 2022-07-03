# -*- coding: utf-8 -*-
# file: train_atepc.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


########################################################################################################################
#                                               ATEPC training script                                                  #
########################################################################################################################
import os

import findfile

from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList, available_checkpoints
from pyabsa.functional import ATEPCConfigManager
from pyabsa import ATEPCCheckpointManager
from pyabsa.functional.dataset import DatasetItem
import torch
import pandas as pd

# atepc_config = ATEPCConfigManager.get_atepc_config_english()

# atepc_config.pretrained_bert = 'microsoft/deberta-v3-large'
# atepc_config.lcf = 'cdm'
# # atepc_config.optimizer = 'adadelta'
# # atepc_config.learning_rate = 0.002
# # atepc_config.hidden_dim = 1024
# # atepc_config.embed_dim = 1024
# atepc_config.model = ATEPCModelList.FAST_LCF_ATEPC
# atepc_config.num_epoch = 1
# dataset_path = DatasetItem('100.CustomDataset')
# or your local dataset: dataset_path = 'your local dataset path'

# for f in findfile.find_cwd_files(['.augment.ignore'] + dataset_path):
#     os.rename(f, f.replace('.augment.ignore', '.augment'))

# aspect_extractor = ATEPCTrainer(config=atepc_config,
#                                 dataset=dataset_path,
#                                 # set checkpoint to train on the checkpoint.
#                                 from_checkpoint='',
#                                 checkpoint_save_mode=1,
#                                 auto_device=True
#                                 ).load_trained_model()

# checkpoint_map = available_checkpoints(from_local=True)
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='/content/gdrive/MyDrive/Tugas_akhir/Rombak_model/PyABSA/demos/aspect_term_extraction/checkpoints/fast_lcf_atepc_100.CustomDataset_cdm_apcacc_83.33_apcf1_69.26_atef1_64.8',
                                                               auto_device=True  # False means load model on CPU
                                                               )

examples = ['But the staff was so nice to us .',
            'But the staff was so horrible to us .',
            r'Not only was the food outstanding , but the little ` perks \' were great .',
            'It took half an hour to get our check , which was perfect since we could sit , have drinks and talk !',
            'It was pleasantly uncrowded , the service was delightful , the garden adorable , '
            'the food -LRB- from appetizers to entrees -RRB- was delectable .',
            'How pretentious and inappropriate for MJ Grill to claim that it provides power lunch and dinners !'
            ]

# df = pd.read_excel('/content/gdrive/MyDrive/Tugas_akhir/Dataset_fix/input_dataset_model/excel/data_pasca_covid.xlsx')
# df = df.drop(columns='Unnamed: 0')
# data_df1 = pd.DataFrame(columns=['id', 'review', 'aspect_dict'])
# for name, grouped in df.groupby('Id'):
#     id = name
#     review = grouped['Text'].iloc[0]
#     aspects = grouped['Aspect']
#     sentiments = grouped['Polarity']
#     aspect_dict = {k.strip(): sentiments.iloc[index]
#                 for index, k in enumerate(aspects)}
#     data_df1 = data_df1.append({'id': id, 'review': review, 'aspect_dict': aspect_dict}, ignore_index=True)
# # new data frame with split value columns
# new = data_df1["review"].str.split(" ", n=1, expand=True)
# # making separate first name column from new data frame
# data_df1["Id"] = new[0]
# # making separate last name column from new data frame
# data_df1["new_review"] = new[1]
# # Dropping old Name columns
# data_df1.drop(columns=["review"], inplace=True)
# data_df1.drop(columns=["Id"], inplace=True)
# review_list = []
# df_tes = data_df1[:50]
# for x in df_tes['new_review']:
#     review_list.append(x)

df = pd.read_csv('/content/gdrive/MyDrive/Tugas_akhir/Dataset_fix/csv_data_total/dbc_borobudur.csv')
df = df.drop(columns='Unnamed: 0')
review_list = []
for x in df['comment']:
    review_list.append(x)

inference_source = ABSADatasetList.Laptop14
atepc_result = aspect_extractor.extract_aspect(inference_source=review_list,
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )
