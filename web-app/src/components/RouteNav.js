import React from 'react';
import {ROUTES} from '../state/Constants';

function RouteNav (props) {
    const {
        activeRoute,
        routeKey,
        routeLabel,
        store,
    } = props;

    const baseClass = 'route-nav-button link';

    return (
        <div
        className={activeRoute === ROUTES[routeKey] ? baseClass + ' active' : baseClass}
        onClick={() => store.setRoute(routeLabel)}>
            {routeLabel}
        </div>
    )
}

export default RouteNav;
