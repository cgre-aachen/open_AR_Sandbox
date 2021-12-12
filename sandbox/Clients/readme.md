This is the place where all the servers go.

A server is a python file that is started manually or automaticvally by a client. 
It exposes functionality over Rest-API endpoints. 

For example each module can have its on server that receives the request to process a depth frame from the kinect, do something with it and send the result back to the client. 

Servers should always be stateless, meaning that with each request all information necessary to handle the request are part of the request itself and the result does not depend on what the server has done before.
If you need the server to be in a certain state because otherwise the request would take too long to compute you have the possibility to expose multiple endpoints for different states: 
For example the gempy server can have a separate endpoint for each model: 
/gempyModule/{ModelID}/compute_frame


Each server needs the following calls implemented:

/{Server}/ready [Get]

returns true if the server is loaded and ready to go (setup complete)
returns false if server is running but not ready (setup not complete)
returns http 404 (or timeout) if server is not running (process not started/ crashed)
returns http 500 if server throws errors 

/{Server}/restart [Post]

Runs Setup again

/{Server}/compute_frame [Get]

This basically runs the update loop once and passes the resulting frame as a response.
All information necessary to compute the frame have to be included in the request (cropped depth frame, etc)




