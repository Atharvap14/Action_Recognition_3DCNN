import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn as nn
import tempfile

classes=['punch', 'pick', 'laugh', 'pour', 'pullup']
# Define the CNN model (you can modify this architecture as needed)
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels = 16, out_channels = 64, kernel_size = 3, stride = 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2),
            nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = (1,3,3), stride = 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(6272, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Load the trained PyTorch model
model_path = 'model_improved_5_epoch.pth'
model = CNNModel(num_classes=5)  # Replace with the appropriate number of classes
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
model.eval()

# Define a function to preprocess input images
def preprocess_image(image):
    
    return image

# Streamlit app
st.title("Video Classification App")
st.sidebar.title("Model Settings")

# Upload a video file
video_file = st.file_uploader("Upload a video file", type=["mp4","avi"])
tfile = tempfile.NamedTemporaryFile(delete=False)
if video_file:
    # Display the uploaded video
    st.video(video_file)
    tfile.write(video_file.read())  

    # Process the video and make predictions
    cap = cv2.VideoCapture(tfile.name)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128))
        frames.append(frame)
    cap.release()

    if len(frames) >= 16:
        st.write("Processing video...")

        # Preprocess frames and make predictions
        clip_length = 16
        predictions = []

        for i in range(0, len(frames) - clip_length + 1):
            clip_frames = frames[i:i + clip_length]
            clip_frames = [preprocess_image(frame) for frame in clip_frames]
            clip_frames = np.stack(clip_frames,axis=0)
            clip_frames = torch.tensor(clip_frames).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                outputs = model(clip_frames.transpose(2,4).float())
                
                _,predicted = torch.max(outputs.data,1)
                predictions.append(predicted.item())

        # Display predictions
        
        predicted_class = max(set(predictions), key=predictions.count)  # Majority vote
        st.write(f"Predicted Class: {classes[predicted_class]}")
    else:
        st.warning("Not enough frames for prediction. Upload a video with at least 16 frames.")

# Optionally, you can add more Streamlit components for user interaction and customization.

