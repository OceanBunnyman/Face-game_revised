# Python 3.12.2
#
# TO RUN THIS FILE:
#
# cd Face-game
# pip install -r requirements.txt
# python main.py
#
# Press 'q' to quit during the game

import cv2
import numpy as np
from deepface import DeepFace
import random
import time
from datetime import datetime, timedelta
from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # hide pygame support msg
environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pygame
import sys

TEST_MODE = True  # enable skipping an emotion by pressing 's'

# Load the pre-trained emotion detection model
model = DeepFace.build_model("Emotion")

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video - if stuck here (e.g. on Linux) try 1 instead of 0
cap = cv2.VideoCapture(0)

# Initialize pygame mixer and sounds
pygame.mixer.init()
effect_success = pygame.mixer.Sound('audio/success.mp3')

emotion_sounds = {
    'angry': pygame.mixer.Sound('audio/angry.mp3'),
    'disgust': pygame.mixer.Sound('audio/disgust.mp3'),
    'fear': pygame.mixer.Sound('audio/fear.mp3'),
    'happy': pygame.mixer.Sound('audio/happy.mp3'),
    'sad': pygame.mixer.Sound('audio/sad.mp3'),
    'surprise': pygame.mixer.Sound('audio/surprise.mp3'),
    'neutral': pygame.mixer.Sound('audio/neutral.mp3')
}

effect_fanfare = pygame.mixer.Sound('audio/tada-fanfare.mp3')
pygame.mixer.music.load('audio/bossa-dreaming-vapor-limiter.mp3')

# Pre-compute images for each emotion, gray background with text
emotion_images = [None] * len(emotion_labels)

for i in range(len(emotion_images)):
    text = emotion_labels[i]
    gray_frame = np.ones((480, 640), dtype=np.uint8) * 128
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (gray_frame.shape[1] - text_size[0]) // 2
    text_y = (gray_frame.shape[0] + text_size[1]) // 2
    cv2.putText(gray_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    emotion_images[i] = gray_frame

# The current target emotion
target_emotion = 'neutral'
target_emotion_image = emotion_images[6]


# Return new target emotion and the corresponding image
def new_target():
    rand = random.randint(0, len(emotion_labels) - 1)
    return [emotion_labels[rand], emotion_images[rand]]


# Runs the detecting phase
def detect():
    # Start playing background music continuously with a fade-in effect
    pygame.mixer.music.play(loops=-1, fade_ms=100)
    done = False

    while not done:
        ret, frame = cap.read()
        if not ret:
            continue  # Skip if frame not captured successfully

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        cv2.imshow('Face game', target_emotion_image)  # Display the current target emotion image by default

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            resized_face = cv2.resize(face_roi, (48, 48))
            rgb_resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
            preds = model.predict(np.expand_dims(rgb_resized_face, axis=0))
            emotion_idx = preds.argmax()
            emotion = emotion_labels[emotion_idx]

            if emotion == target_emotion:
                # Display detected face with emotion label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow('Face game', frame)  # Update the display

                # Stop background music and play the success sound
                pygame.mixer.music.stop()
                emotion_sounds[emotion].play()  # Play the success sound

                # Continuously check if the sound is still playing
                while pygame.mixer.get_busy():
                    # Refresh the display and handle quit event
                    cv2.imshow('Face game', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        done = True
                        break

                if not done:
                    pygame.mixer.music.play(loops=-1, fade_ms=100)  # Resume the background music
                    done = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            done = True  # Allow exiting with 'q'



# Game state for the main loop
points = 0
start_time = datetime.now()
time_limit = 120  # seconds, duration of the game

# START THE GAME
while True:
    ret, frame = cap.read()
    if (not ret):
        continue
    text = "Press Spacebar to begin"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2

    cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Face game', frame)
    if cv2.waitKey(1) & 0xFF == 32:
        break
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

cv2.destroyAllWindows()

# MAIN LOOP
while (datetime.now() < start_time + timedelta(seconds=time_limit)):
    print("Target emotion: ", target_emotion)
    detect()  # detect emotion until goal emotion is detected
    points += 1
    if cv2.waitKey(3000) & 0xFF == ord('q'):
        exit()

    ret = new_target()
    target_emotion = ret[0]
    target_emotion_image = ret[1]

# END SCREEN
text = "You collected " + str(points) + " points!"
print(text)
end_frame = np.ones((480, 640), dtype=np.uint8) * 128
font = cv2.FONT_HERSHEY_SIMPLEX
text_size = cv2.getTextSize(text, font, 1, 2)[0]
text_x = (end_frame.shape[1] - text_size[0]) // 2
text_y = (end_frame.shape[0] + text_size[1]) // 2
cv2.putText(end_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
effect_fanfare.play()
cv2.imshow('Face game', end_frame)
if cv2.waitKey(4000) & 0xFF == ord('q'):
    exit()

# Release the camera and close all windows
pygame.mixer.music.unload()
cap.release()
cv2.destroyAllWindows()