EPOCHS_CNN = {
    'edinburgh_2018': {
        1440: {48: 120, 20: 46},
        5: {48: 28, 20: 119}
        },
    'edinburgh': {
        1440: {100: 59, 50: 58, 20: 999},
        5: {100: 57, 50: 462, 20: 997}
        },
    'leon' : {
        1440: {100: 11, 50: 162},
        5: {100: 15, 50: 52}
        },
    'guadalajara' : {
        1440: {100: 83, 50: 319},
        5: {100: 174, 50: 339}
        }
    }

EPOCHS_LSTM = {
    'edinburgh_2018': {
        5: {48: 4, 20: 16}
        },
    'edinburgh': {
        1440: {100: 23, 50: 30, 20: 31},
        5: {100: 872, 50: 15, 20: 19}
        },
    'leon' : {
        5: {100: 34, 50: 15}
        }
    }

MODEL_RUNS_CNN = {
    'edinburgh_2018': {
        1440: {48: 0, 20: 0},
        5: {48: 0, 20: 0}
        },
    'edinburgh': {
        1440: {100: 0, 50: 0, 20: 0},
        5: {100: 0, 50: 0, 20: 0}
        },
    'leon' : {
        1440: {100: 0, 50: 0},
        5: {100: 0, 50: 0}
        },
    'guadalajara' : {
        1440: {100: 0, 50: 0},
        5: {100: 0, 50: 0}
        }
    }

MODEL_RUNS_LSTM = {
    'edinburgh_2018': {
        5: {48: 0, 20: 0}
        },
    'edinburgh': {
        1440: {100: 0, 50: 0, 20: 0},
        5: {100: 0, 50: 0, 20: 0}
        },
    'leon' : {
        5: {100: 0, 50: 0}
        }
    }