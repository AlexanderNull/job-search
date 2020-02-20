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
jobStore.labelJob = action(function(isMatch) {
    console.log(isMatch ? 'Looks Good!' : 'Another Michigan job?');
    jobStore.nextJob = jobStore.jobs.shift();
    if (jobStore.jobs.length == 0) {
        jobStore.loadJobs();
    }
})

export default jobStore;
