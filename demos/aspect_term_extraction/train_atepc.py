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

atepc_config = ATEPCConfigManager.get_atepc_config_english()

atepc_config.pretrained_bert = 'microsoft/deberta-v3-base'
atepc_config.lcf = 'cdm'
# atepc_config.hidden_dim = 1024
# atepc_config.embed_dim = 1024
atepc_config.model = ATEPCModelList.FAST_LCF_ATEPC
atepc_config.num_epoch = 15
dataset_path = DatasetItem('100.CustomDataset')
# or your local dataset: dataset_path = 'your local dataset path'

# for f in findfile.find_cwd_files(['.augment.ignore'] + dataset_path):
#     os.rename(f, f.replace('.augment.ignore', '.augment'))

# aspect_extractor = ATEPCTrainer(config=atepc_config,
#                                 dataset=dataset_path,
#                                 from_checkpoint='',  # set checkpoint to train on the checkpoint.
#                                 checkpoint_save_mode=1,
#                                 auto_device=True
#                                 ).load_trained_model()

# checkpoint_map = available_checkpoints(from_local=True)
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='/content/gdrive/MyDrive/Tugas_akhir/Rombak_model/PyABSA/demos/aspect_term_extraction/checkpoints/fast_lcf_atepc_100.CustomDataset_cdm_apcacc_87.92_apcf1_80.88_atef1_71.08',
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

inference_source = ABSADatasetList.Laptop14
atepc_result = aspect_extractor.extract_aspect(inference_source=examples,
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )
