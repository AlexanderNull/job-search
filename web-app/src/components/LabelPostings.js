import React from 'react';
import Posting from './Posting';
import NoJobs from './NoJobs';

function LabelPostings (props) {
    const {store} = props;
    const {nextJob, jobs} = store;
    const numberOfJobs = jobs.length;

    console.log('in labels');
    return (
        <div className="postings">
        {(nextJob != null ?
          <Posting store={store} job={nextJob} numberOfJobs={numberOfJobs} /> :
          <NoJobs />)
        }
      </div>
    )
}

export default LabelPostings;
