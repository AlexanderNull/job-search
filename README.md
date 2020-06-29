# job-search
Text classification job posting labeling and suggesting

## Steps to run:
- You'll need to see what your environment's local IP is and update that in App.js
- navigate to web-app and run "npm run build"
- navigate to root and set an environment variable FLASK_APP=app.py
- run "python -m flask run --host=0.0.0.0" if you want this available on your wifi

## Features:
- By default server expects to have access to the google news word embedding model. You can just update the relative path you saved the file if you choose a different save name. Download it at: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit . It's a few GBs so you'll need to grab it on your own.
- If you want however there's a slightly less accurate word embedding layer I trained up on about 2 years worth of Hacker News job posts that is included in this project. Just change the config/pyserver.json setting of "use_google_embedding" to false and it'll use my custom trained layer instead of the large Google layer.
- The api hasn't been super documented yet, but you can adjust most of the tuning parameters via payload properties when posting to the /api/model endpoint. This will eventually be added to an admin console in the web app for ease of use.
- Predictions can be made against the server after model training either individually at /api/model/predict or in bulk at /api/model/predictbulk . Bulk prediction expects both an id and a text property for each item passed in the post body.

## Upcoming:
- Will add a quick admin console for training the model and adjusting parameters/evaluating results
- UI for showing all months of Hacker News Who's Hiring posts, clicking on a month will predict whether each job in selected post is preferred or not and will
filter out all posts predicted as unpreferred.
