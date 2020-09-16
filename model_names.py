from constants_jo import ROOT_DIR

ALL_DATA_DIR = f'{ROOT_DIR}/data/processed'
ROOT_MODEL_DIR = f'{ROOT_DIR}/code/models'

CELL_SIZES = {
    'leon': [50, 100],
    'guadalajara': [50, 100],
    'edinburgh': [100, 50, 20],
    'edinburgh_2018': [48, 20]
              }

### cnn models
INTERVALS = {}
INTERVALS['cnn'] = {
    'leon': [5,1440],
    'guadalajara': [5,1440],
    'edinburgh': [1440, 5],
    'edinburgh_2018': [5, 1440]
                    }

### lstm models
INTERVALS['lstm'] = {
    'leon': [5,],
    'guadalajara': [],
    'edinburgh': [5,1440],
    'edinburgh_2018': [5,]
    }

GRID_DIMS = {
    'leon': {100:(25,40), 50:(48,79)},
    'guadalajara': {100:(16,28), 50:(32,55)},
    'edinburgh': {100:(37, 39), 50:(73,77), 20:(181,191)},
    'edinburgh_2018': {48:(21,20), 20:(51,49)}
    }

SEQ_LENS = {
    'leon': {5: 2},
    'edinburgh': {5: 4, 1440: 3},
    'edinburgh_2018': {5: 2}
    }