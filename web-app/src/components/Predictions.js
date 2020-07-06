import React from 'react';
import {observer} from 'mobx-react';
import '../styles/Predictions.css';

const Predictions = observer(function InnerPredictions (props) {
    const {store} = props;
    const {monthPosts, predictMonth, loadingMonths} = store;

    if (loadingMonths) {
        return <div className="loading">Loading Months</div>
    } else if (predictMonth == null) {
        return <PredictionMonths store={store} />
    } else {
        return <PredictionJobs store={store} />
    }
});

function PredictionMonths (props) {
    const {store} = props;
    const {monthPosts} = store;

    console.log("posts", monthPosts);
    return (
        <div className="prediction-months">
            {monthPosts.map(month => (
                <div className="month link" onClick={() => store.setPredictMonth(month.id)}>{month.title}</div>
            ))}
        </div>
    )
}

const PredictionJobs = observer(function InnnerPredictionJobs (props) {
    const {store} = props;
    const {loadingPredictions, preferredPredictions} = store;

    if (loadingPredictions) {
        return <div className="loading">Loading Predictions</div>
    } else {
        return (
            <div className="prediction-jobs">
                {preferredPredictions.map(job => (
                    <div className="job">{job.text}</div>
                ))}
            </div>
        )
    }
});

export default Predictions;
