import {action, observable} from 'mobx';

const serverUrl = 'http://localhost:5000'

const jobStore = observable({
    nextJob: null,
    jobs: [],
});

// TODO: call server, get next X jobs
jobStore.loadJobs = action(async function () {
    console.log('Grabbing more jobs!');
    const jobsCall = await fetch(`${serverUrl}/api/jobs/unlabeled`);
    const moreJobs = await jobsCall.json();
    if (jobStore.nextJob == null) {
        jobStore.nextJob = moreJobs.shift();
    }
    jobStore.jobs = moreJobs;
});

jobStore.handleKeyDown = action(function(keyEvent) {
    switch (keyEvent.keyCode) {
        case 37:
            // left arrow
            jobStore.labelJob(false);
            keyEvent.stopPropagation();
            break;
        case 39:
            // right arrow
            jobStore.labelJob(true);
            keyEvent.stopPropagation();
            break;
        default:
            // don't care about other keys, ignore
    }
});

// TODO: send label to server
jobStore.labelJob = action(async function(isMatch) {
    const updateCall = await fetch(`${serverUrl}/api/jobs/${jobStore.nextJob.id}`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            'preferred': isMatch,
        }),
    });
    if (updateCall.status === 200) {
        jobStore.nextJob = jobStore.jobs.shift();
        if (jobStore.jobs.length == 0) {
            jobStore.loadJobs();
        }

    } else {
        console.log('Couldn\'t update current job, server call failed');
    }
});

export default jobStore;
