import React from 'react';
import JobText from './JobText';

function Posting (props) {
    const {
        job: {
            date,
            id,
            text,
            by,
        },
        numberOfJobs
    } = props;

    const parsedDate = new Date();
    parsedDate.setTime(date * 1000);

    return (
        <div className="posting">
            <section className="head">
                <div className="links"><a href={`https://news.ycombinator.com/item?id=${id}`}>{by}</a></div>
                <div className="date">{parsedDate.toLocaleDateString()}</div>
                <div className="info">Remaining: {numberOfJobs}</div>
            </section>
            <section className="description">
                <JobText dangerousText={text} />
            </section>
        </div>
    )
}

export default Posting;
