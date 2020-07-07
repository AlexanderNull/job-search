import React from 'react';
import {observer} from 'mobx-react';
import {LineChart, XAxis, YAxis, Line, Tooltip, Legend} from 'recharts';
import {MODEL_CONTROLS} from '../state/Constants';
import '../styles/ModelAdmin.css';

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
        return <div className="loading">Training Model, this may take a while </div>
    } else if (trainingHistory.length > 0) {
        return (
            <div className="charts">
                <LineChart width={700} height={400} data={trainingHistory}>
                    <XAxis dataKey="epoch" interval="preserveStartEnd" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line dataKey="loss" stroke="#fca903" />
                    <Line dataKey="validationLoss" />
                </LineChart>
                <LineChart width={700} height={400} data={trainingHistory}>
                    <XAxis dataKey="epoch" interval="preserveStartEnd" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line dataKey="accuracy" stroke="#fca903" />
                    <Line dataKey="validationAccuracy" />
                </LineChart>
            </div>
        )
    } else {
        return null;
    }
}

export default ModelAdmin;
