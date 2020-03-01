import React from 'react';
import '../styles/App.css';
import {observer} from 'mobx-react';
import Posting from './Posting';
import NoJobs from './NoJobs';
import jobStore from '../state/JobStore';

class App extends React.Component {
  componentDidMount() {
    window.onkeydown = jobStore.handleKeyDown;
    jobStore.loadJobs();
  }

  render() {
    return <WrappableApp />
  }
}

const WrappableApp = observer(function (props) {
  const { nextJob, jobs } = jobStore;
  const numberOfJobs = jobs.length;
    
  return (
    <main className="App" onKeyDown={console.log}>
      <div className="postings">
        {(nextJob != null ?
          <Posting store={jobStore} job={nextJob} numberOfJobs={numberOfJobs} /> :
          <NoJobs />)
        }
      </div>
    </main>
  );
})

export default App;
