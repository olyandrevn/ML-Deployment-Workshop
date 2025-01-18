# ML-Deployment-Workshop

### Project Setup

#### Clone the Repository 

```
git clone <repository_url>
cd <repository_directory>
```
#### Install Dependencies

```
python -m venv workshop
source workshop/bin/activate
pip install -r requirements.txt
```

### Test API

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Test the prediction

```
curl -X POST "http://127.0.0.1:8000/test"
```
### Docker API

Build and run the Docker Image

```
docker build -t fastapi-mnist-app .
docker run -p 8000:8000 fastapi-mnist-app
```
