# -*- coding: utf-8 -*-
# file: __init__.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95

# Copyright (C) 2021. All Rights Reserved.

__version__ = '1.14.8'
__name__ = 'pyabsa'

from termcolor import colored
from update_checker import UpdateChecker

from pyabsa.functional.trainer import APCTrainer, ATEPCTrainer, TCTrainer, AOTCTrainer
from pyabsa.core.apc.models import (APCModelList,
                                    BERTBaselineAPCModelList,
                                    GloVeAPCModelList)
from pyabsa.core.tc.models import (GloVeTCModelList,
                                   BERTTCModelList)
from pyabsa.core.ao_tc.models import (AOGloVeTCModelList,
                                      AOBERTTCModelList)
from pyabsa.core.atepc.models import ATEPCModelList

from pyabsa.functional import (TCCheckpointManager,
                               AOTCCheckpointManager,
                               APCCheckpointManager,
                               ATEPCCheckpointManager,
                               )
from pyabsa.functional.checkpoint.checkpoint_manager import (APCCheckpointManager,
                                                             ATEPCCheckpointManager,
                                                             available_checkpoints)
from pyabsa.functional.dataset import ABSADatasetList, TCDatasetList, AdvTCDatasetList
from pyabsa.functional.config import APCConfigManager
from pyabsa.functional.config import ATEPCConfigManager
from pyabsa.functional.config import TCConfigManager
from pyabsa.functional.config import AOTCConfigManager
from pyabsa.utils.file_utils import check_update_log, validate_datasets_version
from pyabsa.utils.pyabsa_utils import validate_pyabsa_version

# compatible for v1.14.3 and earlier versions
ClassificationDatasetList = TCDatasetList
TextClassifierCheckpointManager = TCCheckpointManager
GloVeClassificationModelList = GloVeTCModelList
BERTClassificationModelList = BERTTCModelList
ClassificationConfigManager = TCConfigManager
TextClassificationTrainer = TCTrainer

validate_pyabsa_version()

validate_datasets_version()

checker = UpdateChecker()
check_result = checker.check(__name__, __version__)

if check_result:
    print(check_result)
    check_update_log()
