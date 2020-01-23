import {action, observable} from 'mobx';

const getJob = (i) => ({
    company: 'pfizer',
    state: 'oregon',
    city: 'portland',
    description: 'Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot. Lorem ipsum but like a lot.',
    title: `Job #${i}`,
}) 

const jobStore = observable({
    nextJob: null,
    jobs: [],
});

// TODO: call server, get next X jobs
jobStore.loadJobs = action(function () {
    console.log('Grabbing more jobs!');
    const newJobs = Array(10).fill(null).map((x, i) => getJob(i));
    if (jobStore.nextJob == null) {
        jobStore.nextJob = newJobs.shift();
    }
    jobStore.jobs = newJobs;
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
