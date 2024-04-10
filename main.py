import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ["Car", "Dragonfly", "Ice cream"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# custom_model_final:
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=128 * 28 * 28, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MyModel().to(device)

model.load_state_dict(torch.load('custom_model_final.pth', map_location=torch.device(device)))
model.eval()

def class_statistics(truth, predictions, threshold):
    predictions_binary = (predictions > threshold).astype(int)
    accuracy = accuracy_score(truth, predictions_binary)
    recall = recall_score(truth, predictions_binary)
    precision = precision_score(truth, predictions_binary)
    f1 = f1_score(truth, predictions_binary)

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1
    }

def calculate_probability(probabilities, class_id, label_arr, class_label): # (probabilities: [n_samples, n_classes], class_id: int, label_arr: [n_samples], class_label: int) -> [(float), (bool)]
    return [(probabilities[i][class_id],
             float(label_arr[i] == class_label)) for i in range(len(label_arr))]

def get_probabilities(array):
    return [probability for probability, _ in array]

def get_truths(arr):
    return [truth for _, truth in arr]

def calculate_statistics(probabilities, threshold):
  truth_list = get_truths(probabilities)
  probabilities_list = get_probabilities(probabilities)

  return class_statistics(np.array(truth_list),
                           np.array(probabilities_list),
                           threshold)

thresholds = [0.8, 0.8, 0.8]

def predict_and_display(image, model, transform):
    model.eval()
    image_transformed = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_transformed)
    probabilities = torch.sigmoid(outputs[0])
    binary_probabilities = probabilities.cpu().numpy()
    return binary_probabilities

@app.route("/predict", methods=['POST'])
def main():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    if file:
        selected_image = Image.open(file.stream).convert('RGB')
        predicted_class = predict_and_display(selected_image, model, transform)
        print(f"{classes[0]}={round(float(predicted_class[0]), 5)} | {classes[1]}={round(float(predicted_class[1]), 5)} | {classes[2]}={round(float(predicted_class[2]), 5)}")
        return f"{classes[0]}={round(float(predicted_class[0]), 5)} | {classes[1]}={round(float(predicted_class[1]), 5)} | {classes[2]}={round(float(predicted_class[2]), 5)}"

    return "Something went wrong", 500

if __name__ == "__main__":
    print("yes")
    app.run()
