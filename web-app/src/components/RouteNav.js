import React from 'react';

function RouteNav (props) {
    const {
        activeRoute,
        routeKey,
        routeLabel,
        store,
    } = props;

    const baseClass = 'route-nav-button';

    return (
        <div
        className={activeRoute === routeKey ? baseClass + ' active' : baseClass}
        onClick={() => store.setRoute(routeLabel)}>
            {routeLabel}
        </div>
    )
}

export default RouteNav;
