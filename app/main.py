from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import torch
from torchvision import transforms, datasets, models
from PIL import Image
import torch.nn as nn

# Initialize FastAPI app
app = FastAPI()

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.model(x)

MODEL_PATH = "app/mnist_resnet_checkpoint.pth"
model = MNISTModel()
device = torch.device("cpu")  # Use CPU explicitly
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.eval()

# Define image preprocessing pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
    transforms.Resize((28, 28)),                 # Resize to 28x28
    transforms.ToTensor(),                       # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))         # Normalize to [-1, 1]
])

# Define a response model
class PredictionResponse(BaseModel):
    predictions: List[float]
    predicted_digit: int

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/test")
def test_api():
    """
    Test the API with MNIST dataset images.
    """
    try:
        # Load MNIST test dataset
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

        # Get a single test sample
        data_iter = iter(test_loader)
        images, labels = next(data_iter)

        # Perform prediction
        with torch.no_grad():
            outputs = model(images)
            predictions = torch.nn.functional.softmax(outputs[0], dim=0).tolist()
            predicted_digit = int(torch.argmax(outputs[0]))

        # Return predictions and the actual label
        return {
            "predictions": predictions,
            "predicted_digit": predicted_digit,
            "actual_label": labels.item()
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict", response_model=PredictionResponse)
def predict_digit(file: UploadFile = File(...)):
    """
    Predict the digit from the uploaded image file.
    """
    try:
        # Read and preprocess the image
        image = Image.open(file.file)
        processed_image = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform prediction on CPU
        with torch.no_grad():
            processed_image = processed_image.to(device)  # Move image to CPU
            outputs = model(processed_image)
            predictions = torch.nn.functional.softmax(outputs[0], dim=0).tolist()
            predicted_digit = int(torch.argmax(outputs[0]))

        # Return predictions and the most likely digit
        return {
            "predictions": predictions,
            "predicted_digit": predicted_digit,
        }
    except Exception as e:
        return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
