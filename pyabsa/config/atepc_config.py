# -*- coding: utf-8 -*-
# file: atepc_config.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import copy

atepc_param_dict_base = {'model_name': "lcf_atepc",
                         'optimizer': "adamw",
                         'learning_rate': 0.00003,
                         'pretrained_bert_name': "bert-base-uncased",
                         'use_dual_bert': False,
                         'use_bert_spc': False,
                         'max_seq_len': 80,
                         'SRD': 3,
                         'use_syntax_based_SRD': False,
                         'lcf': "cdw",
                         'window': "lr",  # unused yet
                         'dropout': 0.5,
                         'l2reg': 0.0001,
                         'num_epoch': 10,
                         'batch_size': 16,
                         'initializer': 'xavier_uniform_',
                         'seed': {1, 2, 3},
                         'embed_dim': 768,
                         'hidden_dim': 768,
                         'polarities_dim': 2,
                         'log_step': 50,
                         'gradient_accumulation_steps': 1,
                         'dynamic_truncate': True,
                         'srd_alignment': True,  # for srd_alignment
                         'evaluate_begin': 0
                         }

atepc_param_dict_english = {'model_name': "lcf_atepc",
                            'optimizer': "adamw",
                            'learning_rate': 0.00002,
                            'pretrained_bert_name': "bert-base-uncased",
                            'use_dual_bert': False,
                            'use_bert_spc': False,
                            'max_seq_len': 80,
                            'SRD': 3,
                            'use_syntax_based_SRD': False,
                            'lcf': "cdw",
                            'window': "lr",
                            'dropout': 0.5,
                            'l2reg': 0.00005,
                            'num_epoch': 10,
                            'batch_size': 16,
                            'initializer': 'xavier_uniform_',
                            'seed': {1, 2, 3},
                            'embed_dim': 768,
                            'hidden_dim': 768,
                            'polarities_dim': 2,
                            'log_step': 50,
                            'gradient_accumulation_steps': 1,
                            'dynamic_truncate': True,
                            'srd_alignment': True,  # for srd_alignment
                            'evaluate_begin': 0
                            }

atepc_param_dict_chinese = {'model_name': "lcf_atepc",
                            'optimizer': "adamw",
                            'learning_rate': 0.00002,
                            'pretrained_bert_name': "bert-base-chinese",
                            'use_dual_bert': False,
                            'use_bert_spc': False,
                            'max_seq_len': 80,
                            'SRD': 3,
                            'use_syntax_based_SRD': False,
                            'lcf': "cdw",
                            'window': "lr",
                            'dropout': 0.5,
                            'l2reg': 0.00005,
                            'num_epoch': 10,
                            'batch_size': 16,
                            'initializer': 'xavier_uniform_',
                            'seed': {1, 2, 3},
                            'embed_dim': 768,
                            'hidden_dim': 768,
                            'polarities_dim': 2,
                            'log_step': 50,
                            'gradient_accumulation_steps': 1,
                            'dynamic_truncate': True,
                            'srd_alignment': True,  # for srd_alignment
                            'evaluate_begin': 0
                            }

atepc_param_dict_multilingual = {'model_name': "lcf_atepc",
                                 'optimizer': "adamw",
                                 'learning_rate': 0.00002,
                                 'pretrained_bert_name': "bert-base-multilingual-uncased",
                                 'use_dual_bert': False,
                                 'use_bert_spc': False,
                                 'max_seq_len': 80,
                                 'SRD': 3,
                                 'use_syntax_based_SRD': False,
                                 'lcf': "cdw",
                                 'window': "lr",
                                 'dropout': 0.5,
                                 'l2reg': 0.00005,
                                 'num_epoch': 10,
                                 'batch_size': 16,
                                 'initializer': 'xavier_uniform_',
                                 'seed': {1, 2, 3},
                                 'embed_dim': 768,
                                 'hidden_dim': 768,
                                 'polarities_dim': 2,
                                 'log_step': 50,
                                 'gradient_accumulation_steps': 1,
                                 'dynamic_truncate': True,
                                 'srd_alignment': True,  # for srd_alignment
                                 'evaluate_begin': 0
                                 }


def get_atepc_param_dict_base():
    return copy.deepcopy(atepc_param_dict_base)


def get_atepc_param_dict_english():
    return copy.deepcopy(atepc_param_dict_english)


def get_atepc_param_dict_chinese():
    return copy.deepcopy(atepc_param_dict_chinese)


def get_atepc_param_dict_multilingual():
    return copy.deepcopy(atepc_param_dict_multilingual)