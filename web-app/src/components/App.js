import React from 'react';
import '../styles/App.css';
import {observer} from 'mobx-react';
import LabelPostings from './LabelPostings';
import RouteNav from './RouteNav';
import Predictions from './Predictions';
import PreferredJobs from './PreferredJobs';
import jobStore from '../state/JobStore';
import {ROUTES} from '../state/JobStore';

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
  const { activeRoute } = jobStore;
  const Page = router(activeRoute, jobStore);
  return (
    <main className="App" onKeyDown={console.log}>
      <nav>{Object.entries(ROUTES).map(([k, v]) => <RouteNav store={jobStore} activeRoute={activeRoute} routeKey={k} routeLabel={v} />)}</nav>
      {Page}
    </main>
  );
});

function router (activeRoute, store) {
  switch (activeRoute) {
    case ROUTES.HOME:
      return (<div>HI</div>)
    case ROUTES.LABEL:
      return <LabelPostings store={store} />
    case ROUTES.PREDICT:
      return <Predictions store={store} />
    case ROUTES.PREFERRED:
      return <PreferredJobs store={store} />
  }
}

export default App;
