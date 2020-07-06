import React from 'react';
import '../styles/App.css';
import {observer} from 'mobx-react';
import LabelPostings from './LabelPostings';
import RouteNav from './RouteNav';
import Predictions from './Predictions';
import PreferredJobs from './PreferredJobs';
import ModelAdmin from './ModelAdmin';
import jobStore from '../state/JobStore';
import modelStore from '../state/ModelStore';
import {ROUTES} from '../state/Constants';

class App extends React.Component {
  render() {
    return <WrappableApp />
  }
}

const WrappableApp = observer(function (props) {
  const { activeRoute } = jobStore;
  const Page = router(activeRoute, jobStore, modelStore);
  return (
    <main className="App">
      <nav>{Object.entries(ROUTES).map(([k, v]) => <RouteNav store={jobStore} activeRoute={activeRoute} routeKey={k} routeLabel={v} />)}</nav>
      {Page}
    </main>
  );
});

function router (activeRoute, jobStore, modelStore) {
  switch (activeRoute) {
    case ROUTES.HOME:
      return (<div>Welcome to your handy job search assistant!</div>)
    case ROUTES.LABEL:
      return <LabelPostings store={jobStore} />
    case ROUTES.PREDICT:
      return <Predictions store={jobStore} />
    case ROUTES.PREFERRED:
      return <PreferredJobs store={jobStore} />
    case ROUTES.ADMIN:
      return <ModelAdmin store={modelStore} />
  }
}

export default App;
