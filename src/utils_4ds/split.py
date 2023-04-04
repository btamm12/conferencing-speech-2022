

from enum import Enum


class Split(Enum):
    TRAIN = 0
    TRAIN_SUBSET = 1
    TRAIN_PSTN = 2
    TRAIN_TENCENT = 3
    TRAIN_NISQA = 4
    TRAIN_IUB = 5
    VAL = 6
    VAL_SUBSET = 7
    VAL_PSTN = 8
    VAL_TENCENT = 9
    VAL_NISQA = 10
    VAL_IUB = 11
    TEST = 12


DEV_SPLITS = [
    *(Split.TRAIN, Split.TRAIN_SUBSET),
    *(Split.TRAIN_PSTN, Split.TRAIN_TENCENT),
    *(Split.TRAIN_NISQA, Split.TRAIN_IUB),
    *(Split.VAL, Split.VAL_SUBSET),
    *(Split.VAL_PSTN, Split.VAL_TENCENT),
    *(Split.VAL_NISQA, Split.VAL_IUB),
]
TRAIN_SUBSET_SPLITS = [
    *(Split.TRAIN_SUBSET, Split.TRAIN_PSTN, Split.TRAIN_TENCENT),
    *(Split.TRAIN_NISQA, Split.TRAIN_IUB),
]
VAL_SUBSET_SPLITS = [
    *(Split.VAL_SUBSET, Split.VAL_PSTN, Split.VAL_TENCENT),
    *(Split.VAL_NISQA, Split.VAL_IUB),
]
SUBSET_SPLITS = TRAIN_SUBSET_SPLITS + VAL_SUBSET_SPLITS
ALL_SPLITS = DEV_SPLITS + [Split.TEST]
