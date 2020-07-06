import React from 'react';
import {observer} from 'mobx-react';
import JobText from './JobText';

const Predictions = observer(function InnerPredictions (props) {
    const {store} = props;
    const {predictMonth, loadingMonths} = store;

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
        const len = preferredPredictions.length;
        return (
            <div className="prediction-jobs">
                {preferredPredictions.map((job, i) => <JobText dangerousText={job.text} shouldSplit={i < (len - 1)} />)}
            </div>
        )
    }
});

export default Predictions;
