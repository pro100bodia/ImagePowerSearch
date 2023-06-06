# ImagePowerSearch

cd app
python3 -m uvicorn controller:app --reload
docker run --name consul -p 8500:8500 consul agent -dev -ui -client=0.0.0.0 -bind=0.0.0.0