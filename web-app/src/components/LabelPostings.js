import React from 'react';
import Posting from './Posting';
import NoJobs from './NoJobs';

class LabelPostings extends React.Component {
  
  componentDidMount() {
    window.onkeydown = this.props.store.handleKeyDown;
  }

  componentWillUnmount() {
    window.onkeydown = null;
  }

  render () {
    const {store} = this.props;
    const {nextJob, jobs} = store;
    const numberOfJobs = jobs.length;

    return (
        <div className="postings">
        {(nextJob != null ?
          <Posting store={store} job={nextJob} numberOfJobs={numberOfJobs} /> :
          <NoJobs />)
        }
      </div>
    )
  }
}

export default LabelPostings;
