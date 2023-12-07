import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.models import load_model
from flask import Flask, Response, render_template, url_for, request, redirect
from flask import Flask

app = Flask(__name__)           

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
def draw_styled_landmarks(image, results):
    #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             #mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             #mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             #) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


actions = np.array(['What', 'Thank you', 'Sorry', 'I love you', 
                    'How are you', 'Hello', 'Good Morning', 
                    'Good afternoon', 'Excuse me', 'Bad'])

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        color = colors[num % len(colors)] 
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), color, -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

model = load_model('action.h5')

@app.route('/video_feed', methods = ['GET', 'POST'])
def video_feed():
    while True:
        predicting = []
        draw_param = request.args.get('draw')
        while True:
            if draw_param == 'True':
                    predds = request.args.get('predds')
                    if predds == 'show':
                        predicting = True
                        draw_param == 'True'
                    else:
                        predicting = False
                        draw_param == 'True'
                    def generate_frames():
                            sequence = []
                            sentence = []
                            predictions = []
                            threshold = 0.5

                            cap = cv2.VideoCapture(0)
                            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                                
                                while cap.isOpened():
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    
                                    image, results = mediapipe_detection(frame, holistic)
                                    draw_styled_landmarks(image, results)
                                    
                                    keypoints = extract_keypoints(results)
                                    sequence.append(keypoints)
                                    sequence = sequence[-30:]
                                    
                                    if predicting:
                                        if len(sequence) == 30:
                                            res = model.predict(np.expand_dims(sequence, axis=0))[0]
                                            predictions.append(np.argmax(res))

                                            if np.unique(predictions[-10:])[0] == np.argmax(res):
                                                if res[np.argmax(res)] > threshold:
                                                    if len(sentence) > 0:
                                                        if actions[np.argmax(res)] != sentence[-1]:
                                                            sentence.append(actions[np.argmax(res)])
                                                    else:
                                                        sentence.append(actions[np.argmax(res)])

                                            if len(sentence) > 5:
                                                sentence = sentence[-5:]

                                            image = prob_viz(res, actions, image, colors)
                                    
                                    cv2.putText(image, ' '.join(sentence), (3, 30), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                                    
                                    ret, buffer = cv2.imencode('.jpg', image)
                                    frame = buffer.tobytes()

                                    # Show result at the upper side of the screen
                                    cv2.putText(image, ' '.join(sentence), (300, 420), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
                                    
                                    yield (b'--frame\r\n'
                                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
            else:
                    predds = request.args.get('predds')
                    if predds == 'show':
                        predicting = True
                    else:
                        predicting = False
                    def generate_frames():
                            sequence = []
                            sentence = []
                            predictions = []
                            threshold = 0.5
                            prediction_duration = 0
                            last_prediction_time = time.time()
                            result_disappeared_time = None

                            cap = cv2.VideoCapture(0)
                            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                                prediction_start_time = None
                                prev_prediction = None
                                current_prediction = ""
                                
                                while cap.isOpened():
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    
                                    image, results = mediapipe_detection(frame, holistic)
                                    #draw_styled_landmarks(image, results)
                                    
                                    keypoints = extract_keypoints(results)
                                    sequence.append(keypoints)
                                    sequence = sequence[-30:]
                                    
                                    res = None
                                    if predicting:
                                        # Check if the result has disappeared and it's time to make another prediction
                                        if result_disappeared_time is not None and time.time() - result_disappeared_time >= 2:
                                            result_disappeared_time = None  # Reset the result disappeared time
                                            current_prediction = ""  # Reset the current prediction

                                        if len(sequence) == 30 and result_disappeared_time is None:
                                            res = model.predict(np.expand_dims(sequence, axis=0))[0]
                                            predictions.append(np.argmax(res))

                                            if np.unique(predictions[-10:])[0] == np.argmax(res):
                                                if res[np.argmax(res)] > threshold:
                                                    current_prediction = actions[np.argmax(res)]

                                            if len(sentence) > 5:
                                                sentence = sentence[-5:]

                                            if current_prediction:
                                                result_disappeared_time = time.time()  # Set the result disappeared time
                                    
                                    text = ' '.join([current_prediction] if predicting else [])
                                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                                    text_x = (image.shape[1] - text_size[0]) // 2
                                    text_y = image.shape[0] - 100

                                    cv2.putText(image, text, (text_x, text_y), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                                    
                                    ret, buffer = cv2.imencode('.jpg', image)
                                    frame = buffer.tobytes()

                                    
                                    yield (b'--frame\r\n'
                                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('camerapage'))
    return render_template('index.html')
         
@app.route("/camerapage")
def camerapage():
    return render_template('opencam.html', video_urlshow=url_for('video_feed'))

if __name__ == "__main__":
    app.run(debug=True)

