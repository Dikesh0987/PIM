import cv2
from skimage.transform import resize
import tkinter as tk
from tkinter import ttk
from threading import Thread
import pickle  # Import the pickle module

# Load the trained model with error handling
try:
    model = pickle.load(open('./model.p', 'rb'))
except FileNotFoundError:
    print("Error: The model file 'model.p' does not exist.")
    exit()

# Define a function to preprocess live images
def preprocess_live_image(image):
    # Resize the image to match the training data's size
    image = resize(image, (15, 15))
    # Flatten the image
    image = image.flatten()
    return image

camera = cv2.VideoCapture(0)

# Define a function to update the GUI with the prediction
def update_prediction():
    while capture_started:
        ret, frame = camera.read()

        if not ret:
            print("Failed to capture frame.")
            break

        # Preprocess the live image
        preprocessed_image = preprocess_live_image(frame)

        # Make predictions using the loaded model
        prediction = model.predict([preprocessed_image])

        # Map the prediction to the corresponding category
        categories = ['Healthy', 'Not_Healthy']
        predicted_category = categories[prediction[0]]

        # Update the label in the GUI with the prediction
        prediction_label.config(text=f'Prediction: {predicted_category}')

        # Display the live image
        cv2.imshow('Live Disease Finder', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy the OpenCV window
    camera.release()
    cv2.destroyAllWindows()

# Create a Tkinter window
root = tk.Tk()
root.title("Disease Detection")

# Create a label to display the prediction
prediction_label = ttk.Label(root, text="Prediction: -", font=("Helvetica", 16))
prediction_label.pack(pady=10)

# Create buttons to start and stop image capture
capture_started = False

def start_capture():
    global capture_started
    if not capture_started:
        capture_started = True
        capture_thread = Thread(target=update_prediction)
        capture_thread.start()

start_button = ttk.Button(root, text="Start Capture", command=start_capture)
start_button.pack(pady=10)

def stop_capture():
    global capture_started
    capture_started = False

stop_button = ttk.Button(root, text="Stop Capture", command=stop_capture)
stop_button.pack(pady=10)

root.mainloop()
