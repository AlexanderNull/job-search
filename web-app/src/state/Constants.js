const SERVER_URL = 'http://localhost:5000';

const ROUTES = {
    HOME: 'Home',
    LABEL: 'Label Older Jobs',
    PREDICT: 'Predict Jobs',
    PREFERRED: 'Previously Preferred',
    ADMIN: 'Adjust/Train Model',
};

const MODEL_CONTROLS = {
    learningRate: {
        label: 'Learning rate',
        valueProcessor: parseFloat,
    },
    epochs: {
        label: 'Epochs',
        valueProcessor: parseInt,
    },
    dropout: {
        label: 'Dropout',
        valueProcessor: parseFloat,
    },
    batchSize: {
        label: 'Batch size',
        valueProcessor: parseInt,
    },
    maxSequenceLength: {
        label: 'Sequence length token clipping',
        valueProcessor: parseInt,
    },
    testSplit: {
        label: 'Percent for test validation',
        valueProcessor: parseFloat,
    },
    devSplit: {
        label: 'Percent for dev-test validation',
        valueProcessor: parseFloat,
    },
    lstmUnits: {
        label: 'Units in each LSTM layer',
        valueProcessor: parseInt,
    },
    lstmLayers: {
        label: 'Number of LSTM layers',
        valueProcessor: parseInt,
    },
};

export {
    SERVER_URL,
    ROUTES,
    MODEL_CONTROLS,
}
