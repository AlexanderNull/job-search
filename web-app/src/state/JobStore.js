import {action, observable} from 'mobx';

const serverUrl = 'http://localhost:5000'

const jobStore = observable({
    nextJob: null,
    prevJobs: [],
    jobs: [],
});

// TODO: need a loader here, pulling everything from a month takes a minute or two
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
        case 8: // backspace
        case 46: // delete
            jobStore.goBack();
            keyEvent.preventDefault();
            keyEvent.stopPropagation();
            break;
        default:
            // don't care about other keys, ignore
    }
});

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
        jobStore.prevJobs.push(jobStore.nextJob);
        jobStore.nextJob = jobStore.jobs.shift();
        if (jobStore.jobs.length == 0) {
            jobStore.loadJobs();
        }

    } else {
        console.log('Couldn\'t update current job, server call failed');
    }
});

// Label some of these jobs too fast, need a way to go back to a previously labeled job
jobStore.goBack = action(function () {
    if (jobStore.prevJobs.length > 0) {
        jobStore.jobs.unshift(jobStore.nextJob);
        jobStore.nextJob = jobStore.prevJobs.pop();
    }
});

export default jobStore;
