import React from 'react';

// TODO: handle keydown event here
function Posting(props) {
    const {
        job: {
            company,
            state,
            city,
            description,
            title,
        },
    } = props;

    return (
        <div className="posting">
            <section className="head">
                <div className="title">{title}</div>
                <div className="company">{company}</div>
            </section>
            <section className="description">{description}</section>
            <section className="location">
                <div className="state">{state}</div>
                <div className="city">{city}</div>
            </section>
        </div>
    )
}

export default Posting;
