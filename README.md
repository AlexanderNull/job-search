# job-search
Text classification job posting labeling and suggesting.
- Load up the app and start labeling historical job posts by pressing either Left key (if you're not interested in the job), or Right key (if you are interested in the job).
- Navigate to the Train Model UI to tweak hyperparameters and train a model based on the jobs you've labeled. The number of jobs you'll need labeled will differ based on your preferences, but 2-3 months worth of labeled jobs is a good rough spot.
- Now go to the Predict Jobs UI and select a month to feed through the model, sit back and receive a list of curated job postings that your model thinks you'd be interested in!

![Model training UI](/examples/adminConsole.png?raw=true "Train your model with ease!")

## Steps to run:
- You'll need to see what your environment's local IP is and update that in App.js
- navigate to web-app and run "npm run build"
- navigate to root and set an environment variable FLASK_APP=app.py
- run "python -m flask run --host=0.0.0.0" if you want this available on your wifi

## Features:
- *NEW UI* for tweaking hyperparameters of the model. Manually calling the endpoint is still a bit more flexible, but most pertinent hyperparameters are available in
the admin UI. Training in UI will also provide Loss and Accuracy charts so you can see how the model performed over epochs.
- *NEW UI* for using trained model to predict all jobs you'd be interested in within a chosen month of Who's Hiring posts.
- By default server expects to have access to the google news word embedding model. You can just update the relative path you saved the file if you choose a different save name. Download it at: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit . It's a few GBs so you'll need to grab it on your own.
- If you want however there's a slightly less accurate word embedding layer I trained up on about 2 years worth of Hacker News job posts that is included in this project. Just change the config/pyserver.json setting of "use_google_embedding" to false and it'll use my custom trained layer instead of the large Google layer.
- The api hasn't been super documented yet, but you can adjust most of the tuning parameters via payload properties when posting to the /api/model endpoint. This will eventually be added to an admin console in the web app for ease of use.
- Predictions can be made against the server after model training either individually at /api/model/predict or in bulk at /api/model/predictbulk . Bulk prediction expects both an id and a text property for each item passed in the post body.

## Upcoming:
- Caching, should probably give HN a bit of a break here.
- New interface for reviewing previously labeled posts, good way to review companies you've already expressed interest in.
- Ability to label predicted posts, maybe the model got something wrong?
- More styling, especially around the charts in admin console.
- More hyperparameter control in admin UI, more control over shape of model, and some more data analysis information such as easy lookups for what your data distribution looks like.
