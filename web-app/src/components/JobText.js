import React from 'react';
import '../styles/JobText.css';

// Definitely not going to parse out all of this by hand
// Definitely not going to trust "Hacker" News with having safe input
// Stick to textarea to avoid XSS issues
// TODO: maybe handle the links
function JobText (props) {
    const {
        dangerousText,
        shouldSplit = false,
    } = props;

    var decoder = document.createElement('textarea');
    decoder.innerHTML = dangerousText;
    const elements = [
        (<div className="job-text">
            {decoder.value.split('<p>').map(t => <p>{t}</p>)}
        </div>)
    ];

    if (shouldSplit)
        elements.push(<hr />);

    return elements;
}

export default JobText;
