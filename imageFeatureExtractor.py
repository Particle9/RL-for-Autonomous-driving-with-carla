import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define a custom ResNet feature extractor
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, input_size=(224, 224)):
        super(ResNetFeatureExtractor, self).__init__()
        # Load the ResNet model without the final fully connected layer
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        
        # Store the input size
        self.input_size = input_size
        
        # Define preprocessing transformations with the specified input size
        self.preprocess = transforms.Compose([
            transforms.Resize(self.input_size),          # Resize to the specified input size
            transforms.ToTensor(),                      # Convert image to tensor
            transforms.Normalize(                       # Normalize with ImageNet mean and std
                mean=[0.485, 0.456, 0.406],             # Mean values for R, G, B channels
                std=[0.229, 0.224, 0.225]               # Standard deviations for R, G, B channels
            ),
        ])
        
        self.f1 = nn.Linear(2048, 1024)
        self.f2 = nn.Linear(1024, 512)

    def forward(self, x):
        # Forward pass through the ResNet model without the final FC layer
        x = self.resnet(x)
        x = torch.flatten(x, 1)  # Flatten the output
        x = self.f1(x)
        x = self.f2(x)
        return x

    def extract_features_from_image(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0)  # Add batch dimension
        return self.forward(image_tensor)

    def preprocess_image(self, image):
        image_tensor = self.preprocess(image).unsqueeze(0)
        return image_tensor