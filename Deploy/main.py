import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template  # Fixed 'flask' typo
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import face_recognition
import cv2
import numpy as np
import imageio
import tensorflow as tf  # Added import for tf
 
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
SEQ_LENGTH = 20
 
@app.route('/')
def upload_form():
    return render_template('upload.html')
 
@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_video filename: ' + filename)
        flash('Video is successfully uploaded, Check it out below!')  # Fixed typo
        return render_template('upload.html', filename=filename)
 
def crop_face_center(frame):
    """Extract and crop the face from the frame"""
    face_locations = face_recognition.face_locations(frame)
    if not face_locations:
        return None
   
    # Use the first face found
    top, right, bottom, left = face_locations[0]
    face_image = frame[top:bottom, left:right]
    return face_image
 
def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(total_frames/SEQ_LENGTH), 1)
    frames = []
    try:
        for frame_cntr in range(SEQ_LENGTH):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_cntr*skip_frames_window)
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_face_center(frame)
            if frame is None:
                continue
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2,1,0]]  # BGR to RGB
            frames.append(frame)
            if len(frames) == max_frames:
                break
        print("Completed Extracting Frames Successfully! ;)")
    finally:
        cap.release()
    return np.array(frames)
 
@app.route('/display/<filename>')
def display_video(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
@app.route('/predict/<filename>')
def sequence_prediction(filename):  # Fixed function name typo
    sequence_model = load_model('./models/inceptionNet_model.h5')
    class_vocab = ['FAKE', 'REAL']
    frames = load_video('static/uploads/' + filename)
   
    if len(frames) == 0:
        return render_template('upload.html', filename=filename, prediction="No faces detected")
   
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]  # Fixed variable name typo
    pred = probabilities.argmax()
    return render_template('upload.html', filename=filename, prediction=class_vocab[pred])
 
def prepare_single_video(frames):
    print("Preparing Frames")
    # Get InceptionV3 model for feature extraction (without the classification layer)
    feature_extractor = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
   
    # Normalize frames
    frames = frames / 255.0
   
    # If we don't have enough frames, we'll pad the sequence
    if len(frames) < SEQ_LENGTH:
        # Calculate how many frames we need to pad
        pad_length = SEQ_LENGTH - len(frames)
       
        # If we have at least one frame, we'll duplicate the last frame
        if len(frames) > 0:
            padding = np.repeat(frames[-1:], pad_length, axis=0)
        else:
            # If no frames were extracted, create dummy frames
            padding = np.zeros((pad_length, IMG_SIZE, IMG_SIZE, 3))
       
        frames = np.concatenate([frames, padding], axis=0)
   
    # Extract features using the feature extractor
    frame_features = np.zeros((1, SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    mask = np.zeros((1, SEQ_LENGTH), dtype="bool")
   
    # Extract features for each frame
    for i, frame in enumerate(frames[:SEQ_LENGTH]):
        frame = np.expand_dims(frame, axis=0)
        frame_features[0, i, :] = feature_extractor.predict(frame, verbose=0)
        mask[0, i] = True  # Mark this frame as valid
   
    return frame_features, mask
 
if __name__ == "__main__":
    app.run(debug=True)