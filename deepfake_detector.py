import tensorflow as tf
from tensorflow.keras.applications import Xception  # type:ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions  # type: ignore
import numpy as np
import cv2
import os

class DeepfakeDetector:
    def __init__(self):
        # Load the pre-trained Xception model from Keras
        self.model = Xception(weights='imagenet')
        print("Xception model loaded successfully.")

    def process_image(self, img_path):
        """Process an image and make predictions using the Xception model."""
        img = image.load_img(img_path, target_size=(299, 299))  # Resize image to 299x299 as required by Xception
        x = image.img_to_array(img)  # Convert image to numpy array
        x = np.expand_dims(x, axis=0)  # Create batch axis
        x = preprocess_input(x)  # Preprocess input (scaling, etc.)
        return x

    def predict_image(self, img_path):
        """Predict if the image is real or fake using the pre-trained Xception model."""
        # Process image
        processed_img = self.process_image(img_path)
        # Predict
        predictions = self.model.predict(processed_img)
        # Decode the predictions (optional, for checking what class is predicted)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        print(f"Predictions for {img_path}: {decoded_predictions}")
        
        # You can check the top class, here assuming that 'fake' class maps to a low probability
        # Modify based on your own classification method or fine-tuned model
        if decoded_predictions[0][1] == 'fake':  # Replace 'fake' with correct class for fake if fine-tuned
            return {"status": "fake", "confidence": decoded_predictions[0][2]}
        else:
            return {"status": "real", "confidence": decoded_predictions[0][2]}

    def process_video(self, video_path):
        """Process a video file and detect deepfake frames."""
        cap = cv2.VideoCapture(video_path)
    
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        total_frames = 0
        fake_frames = 0
        real_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame processing (e.g., deepfake detection)
            total_frames += 1
            
            # Example: Update fake/real frame counters
            if self.is_fake(frame):
                fake_frames += 1
            else:
                real_frames += 1

        cap.release()
        
        # Calculate fake percentage
        fake_percentage = (fake_frames / total_frames) * 100 if total_frames else 0
        return {'total_frames': total_frames, 'fake_frames': fake_frames, 'real_frames': real_frames, 'fake_percentage': fake_percentage}

    def is_fake(self, frame):
        """Detect if a frame is fake using the Xception model."""
        # Preprocess the frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame from BGR to RGB
        img = cv2.resize(img, (299, 299))  # Resize the image to 299x299 as required by Xception
        x = np.expand_dims(img, axis=0)  # Add batch dimension
        x = preprocess_input(x)  # Preprocess the image as Xception requires

        # Get predictions from the model
        predictions = self.model.predict(x)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # If the top prediction corresponds to a 'fake' class (you'll need to identify the fake class)
        # This assumes the first element in `decoded_predictions` is 'real' or 'fake' based on fine-tuning
        if decoded_predictions[0][1] == 'fake':  # Modify with actual 'fake' class name
            return True
        else:
            return False


# Example usage
if __name__ == "__main__":
    # Instantiate the detector
    detector = DeepfakeDetector()

    # Detect deepfake in a video
    video_path = "A:/DeepFakeDetectionTool/Deepfake_Detection_Tool/datasets/Celeb-DF-v2/Celeb-synthesis/id0_id9_0001.mp4"
    video_result = detector.process_video(video_path)  # Replace with your video path
    print(f"Video detection result: {video_result}")
