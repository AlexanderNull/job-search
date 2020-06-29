import React from 'react';

function Predictions (props) {
    const {store} = props;
    const {monthPosts, predictMonth} = props;

    if (predictMonth == null) {
        return <PredictionMonths store={store} />
    }
}

function PredictionMonths (props) {
    const {store} = props;
    const {monthPosts} = store;

    return (
        <div className="prediction-months">
            {monthPosts.map(month => (
                <div>
            ))}
        </div>
    )
}

export default Predictions;
