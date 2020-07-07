import {action, observable} from 'mobx';
import {SERVER_URL} from './Constants';

const defaultVals = {
    maxSequenceLength: 200,
    batchSize: 32,
    epochs: 25,
    learningRate: 0.001,
    dropout: 0.2,
    testSplit: 0.2,
    devSplit: 0.2,
    lstmUnits: 32,
    lstmLayers: 2,
    trainingModel: false,
    trainingHistory: {},
};

const modelStore = observable(defaultVals);

// Swear there was a time with mobx where you had to declare all your observables at the start anyways so this ain't quite duplicated code
modelStore.resetModelParameters = action(function () {
    Object.entries(defaultVals).foreach(([key, value]) => modelStore[key] = value);
});

modelStore.trainModel = action(async function () {
    modelStore.trainingModel = true;
    const modelResponse = await fetch(`${SERVER_URL}/api/model`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            max_sequence_length: modelStore.maxSequenceLength,
            batch_size: modelStore.batchSize,
            epochs: modelStore.epochs,
            dropout: modelStore.dropout,
            lstm_layers: modelStore.lstmLayers,
            lstm_units: modelStore.lstmUnits,
            test_split: modelStore.testSplit,
            dev_split: modelStore.devSplit,
            learning_rate: modelStore.learningRate,
        }),
    });
    if (modelResponse.status === 200) {
        const results = await modelResponse.json();
        console.log("Results: ", results);
        modelStore.trainingHistory = formatChartData(results.history);
        modelStore.trainingModel = false;
    } else {
        console.log('Something broke: ', modelResponse);
        modelStore.trainingModel = false;
    }
});

modelStore.updateControl = action(function (controlProp, newValue) {
    modelStore[controlProp] = newValue;
});

// Assumption is that each data type has the same number of epochs
function formatChartData (history) {
    return history.accuracy.map((h, i) => ({
        epoch: i + 1,
        accuracy: history.accuracy[i],
        loss: history.loss[i],
        validationAccuracy: history.val_accuracy[i],
        validationLoss: history.val_loss[i],
    }));
}

export default modelStore;
