import React from 'react';
import {observer} from 'mobx-react';
import {LineChart, XAxis, YAxis, Line, Tooltip} from 'recharts';
import {MODEL_CONTROLS} from '../state/Constants';

const ModelAdmin = observer(function InnerModelAdmin (props) {
    const {store} = props;
    const {
        trainingHistory,
        trainingModel,
    } = store;

    return (
        <div className="admin-shell">
            <section className="controls">
                {Object.keys(MODEL_CONTROLS).map(control => (
                    <Control
                        label={MODEL_CONTROLS[control].label}
                        name={control}
                        value={store[control]}
                        onChange={(e) => store.updateControl(control, MODEL_CONTROLS[control].valueProcessor(e.target.value))}
                    />
                ))}
            </section>
            <section className="buttons">
                <button onClick={store.trainModel}>Train Model</button>
                <button onClick={store.resetModelParams}>Reset Params</button>
            </section>
            <section className="training-chart">
                <TrainingHistory trainingHistory={trainingHistory} trainingModel={trainingModel} />
            </section>
        </div>
    )
});

function Control (props) {
    const {
        label,
        name,
        value,
        onChange,
    } = props;

    return (
        <div className="control">
            <label>
                {label}
                <input type="number" name={name} value={value} onChange={onChange} />
            </label>
        </div>
    );
}

function TrainingHistory (props) {
    const {trainingHistory, trainingModel} = props;

    if (trainingModel) {
        return <div className="loading" />
    } else if (trainingHistory.length > 0) {
        return (
            <div className="charts">
                <LineChart width={500} height={300} data={trainingHistory}>
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip />
                    <Line dataKey="loss" />
                    <Line dataKey="validationLoss" />
                </LineChart>
                <LineChart width={500} height={300} data={trainingHistory}>
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip />
                    <Line dataKey="accuracy" />
                    <Line dataKey="validationAccuracy" />
                </LineChart>
            </div>
        )
    } else {
        return null;
    }
}

export default ModelAdmin;
