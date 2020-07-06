import React from 'react';
import {observer} from 'mobx-react';
import Posting from './Posting';
import NoJobs from './NoJobs';

const LabelPostings = observer(class LabelPostings extends React.Component {
  
  componentDidMount() {
    window.onkeydown = this.props.store.handleKeyDown;
  }

  componentWillUnmount() {
    window.onkeydown = null;
  }

  render () {
    const {store} = this.props;
    const {nextJob, jobs, loadingJobs} = store;
    const numberOfJobs = jobs.length;

    return (
        <div className="postings">
        {(loadingJobs ?
          <div className="loading">Loading jobs</div> :
          nextJob != null ?
            <Posting store={store} job={nextJob} numberOfJobs={numberOfJobs} /> :
            <NoJobs />)
        }
      </div>
    )
  }
});

export default LabelPostings;
