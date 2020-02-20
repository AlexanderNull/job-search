import React from 'react';

function Posting (props) {
    const {
        job: {
            date,
            id,
            text,
            by,
        },
    } = props;

    const parsedDate = new Date();
    parsedDate.setTime(date * 1000);

    return (
        <div className="posting">
            <section className="head">
                <div className="title"><a href={`https://news.ycombinator.com/item?id=${id}`}>{by}</a></div>
                <div className="company">{parsedDate.toLocaleDateString()}</div>
            </section>
            <section className="description">
                <JobText dangerousText={text} />
            </section>
        </div>
    )
}

// Definitely not going to parse out all of this by hand
// Definitely not going to trust "Hacker" News with having safe input
// Stick to textarea to avoid XSS issues
// TODO: maybe handle the links
function JobText (props) {
    var decoder = document.createElement('textarea');
    decoder.innerHTML = props.dangerousText;
    return (
        <div>
            {decoder.value.split('<p>').map(t => <p>{t}</p>)}
        </div>
    )
}

export default Posting;
