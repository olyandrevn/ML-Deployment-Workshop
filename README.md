# ML-Deployment-Workshop

<img width="876" alt="Screenshot 2025-01-18 at 11 42 37 AM" src="https://github.com/user-attachments/assets/35fe787c-e31a-4588-a2b4-88f773eea97e" />

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
<img width="1190" alt="Screenshot 2025-01-18 at 1 37 54 PM" src="https://github.com/user-attachments/assets/c48bd87f-1d12-44a9-9232-766d50b3f145" />
