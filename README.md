# ImagePowerSearch

Computer vision powered image search. The application has two major endpoints: image upload and image search. During the 
image upload the tasks of object detection, faces recognition and image captioning are performed and the retrieved 
information, together with an image, is uploaded to the database. The search is a simple key-words matching.  

###Steps to run:
1) Create database schema with postgres/scheme.sql
2) Launch Consul configuration service using `docker run --name consul -p 8500:8500 consul agent -dev -ui -client=0.0.0.0 -bind=0.0.0.0`
3) To launch an application server `cd` to the `app` folder and run the following command `python3 -m uvicorn controller:app --reload`