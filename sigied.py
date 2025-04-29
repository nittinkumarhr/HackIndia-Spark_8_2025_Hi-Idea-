import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

class ComprehensiveASLRecognizer:
    def __init__(self, model_path='comprehensive_asl_model.h5'):
        # MediaPipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Enhanced hand detection configuration
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support multiple hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )

        # Comprehensive Gesture Classes
        self.gesture_classes = {
            # Basic Communication Gestures
            0: {'name': 'Hello', 'description': 'Welcoming gesture', 'complexity': 'Low'},
            1: {'name': 'Goodbye', 'description': 'Farewell wave', 'complexity': 'Low'},
            2: {'name': 'Please', 'description': 'Polite request sign', 'complexity': 'Medium'},
            3: {'name': 'Thank You', 'description': 'Gratitude expression', 'complexity': 'Low'},
            4: {'name': 'Yes', 'description': 'Affirmative nod', 'complexity': 'Low'},
            5: {'name': 'No', 'description': 'Negative head shake', 'complexity': 'Low'},
            6: {'name': 'Learn', 'description': 'Finger to brain', 'complexity': 'Medium'},
            7: {'name': 'Help', 'description': 'Hand clasping', 'complexity': 'Medium'},
            8: {'name': 'I Love You', 'description': 'Three-finger sign', 'complexity': 'Low'},
            
            # Emotional and Descriptive Gestures
            45: {'name': 'Good', 'description': 'Thumbs up gesture', 'complexity': 'Low'},
            46: {'name': 'Bad', 'description': 'Thumbs down gesture', 'complexity': 'Low'},
            47: {'name': 'Sorry', 'description': 'Apologetic sign', 'complexity': 'Low'},
            48: {'name': 'Friend', 'description': 'Interlocking fingers', 'complexity': 'Medium'},
            49: {'name': 'Family', 'description': 'Family hand gesture', 'complexity': 'Medium'},
            50: {'name': 'Name', 'description': 'Spelling out name', 'complexity': 'Medium'},
            
            # Alphabet (A-Z)
            9: {'name': 'A', 'description': 'Letter A in ASL', 'complexity': 'Low'},
            10: {'name': 'B', 'description': 'Letter B in ASL', 'complexity': 'Low'},
            11: {'name': 'C', 'description': 'Letter C in ASL', 'complexity': 'Low'},
            12: {'name': 'D', 'description': 'Letter D in ASL', 'complexity': 'Low'},
            13: {'name': 'E', 'description': 'Letter E in ASL', 'complexity': 'Low'},
            14: {'name': 'F', 'description': 'Letter F in ASL', 'complexity': 'Low'},
            15: {'name': 'G', 'description': 'Letter G in ASL', 'complexity': 'Low'},
            16: {'name': 'H', 'description': 'Letter H in ASL', 'complexity': 'Low'},
            17: {'name': 'I', 'description': 'Letter I in ASL', 'complexity': 'Low'},
            18: {'name': 'J', 'description': 'Letter J in ASL', 'complexity': 'Low'},
            19: {'name': 'K', 'description': 'Letter K in ASL', 'complexity': 'Low'},
            20: {'name': 'L', 'description': 'Letter L in ASL', 'complexity': 'Low'},
            21: {'name': 'M', 'description': 'Letter M in ASL', 'complexity': 'Low'},
            22: {'name': 'N', 'description': 'Letter N in ASL', 'complexity': 'Low'},
            23: {'name': 'O', 'description': 'Letter O in ASL', 'complexity': 'Low'},
            24: {'name': 'P', 'description': 'Letter P in ASL', 'complexity': 'Low'},
            25: {'name': 'Q', 'description': 'Letter Q in ASL', 'complexity': 'Low'},
            26: {'name': 'R', 'description': 'Letter R in ASL', 'complexity': 'Low'},
            27: {'name': 'S', 'description': 'Letter S in ASL', 'complexity': 'Low'},
            28: {'name': 'T', 'description': 'Letter T in ASL', 'complexity': 'Low'},
            29: {'name': 'U', 'description': 'Letter U in ASL', 'complexity': 'Low'},
            30: {'name': 'V', 'description': 'Letter V in ASL', 'complexity': 'Low'},
            31: {'name': 'W', 'description': 'Letter W in ASL', 'complexity': 'Low'},
            32: {'name': 'X', 'description': 'Letter X in ASL', 'complexity': 'Low'},
            33: {'name': 'Y', 'description': 'Letter Y in ASL', 'complexity': 'Low'},
            34: {'name': 'Z', 'description': 'Letter Z in ASL', 'complexity': 'Low'},
            
            # Numbers
            35: {'name': '0', 'description': 'Number 0 in ASL', 'complexity': 'Low'},
            36: {'name': '1', 'description': 'Number 1 in ASL', 'complexity': 'Low'},
            37: {'name': '2', 'description': 'Number 2 in ASL', 'complexity': 'Low'},
            38: {'name': '3', 'description': 'Number 3 in ASL', 'complexity': 'Low'},
            39: {'name': '4', 'description': 'Number 4 in ASL', 'complexity': 'Low'},
            40: {'name': '5', 'description': 'Number 5 in ASL', 'complexity': 'Low'},
            41: {'name': '6', 'description': 'Number 6 in ASL', 'complexity': 'Low'},
            42: {'name': '7', 'description': 'Number 7 in ASL', 'complexity': 'Low'},
            43: {'name': '8', 'description': 'Number 8 in ASL', 'complexity': 'Low'},
            44: {'name': '9', 'description': 'Number 9 in ASL', 'complexity': 'Low'},
            
            # Advanced Conversation Gestures
            51: {'name': 'Why', 'description': 'Questioning gesture', 'complexity': 'Medium'},
            52: {'name': 'When', 'description': 'Time-related gesture', 'complexity': 'Medium'},
            53: {'name': 'Where', 'description': 'Location inquiry sign', 'complexity': 'Medium'},
            54: {'name': 'How', 'description': 'Method or manner sign', 'complexity': 'Medium'},
            
            # Emotional Expressions
            55: {'name': 'Happy', 'description': 'Happiness expression', 'complexity': 'Low'},
            56: {'name': 'Sad', 'description': 'Sadness expression', 'complexity': 'Low'},
            57: {'name': 'Angry', 'description': 'Anger expression', 'complexity': 'Medium'},
            58: {'name': 'Surprised', 'description': 'Surprise gesture', 'complexity': 'Low'},
            
            # Practical Gestures
            59: {'name': 'Drink', 'description': 'Drinking motion', 'complexity': 'Low'},
            60: {'name': 'Eat', 'description': 'Eating gesture', 'complexity': 'Low'},
            61: {'name': 'Sleep', 'description': 'Sleeping sign', 'complexity': 'Low'},
            62: {'name': 'Weather', 'description': 'Weather-related sign', 'complexity': 'Medium'}
        }

        self.model_path = model_path
        self.model = self.create_advanced_model()

    def create_advanced_model(self):
        """
        Create a sophisticated neural network for gesture recognition
        """
        model = Sequential([
            Dense(256, activation='relu', input_shape=(63,)),  # Expanded input to capture more landmark details
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(len(self.gesture_classes), activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def extract_enhanced_landmarks(self, hand_landmarks):
        """
        Advanced landmark extraction with additional feature engineering
        """
        landmarks = []
        
        # Extract x, y, z coordinates for each landmark
        for landmark in hand_landmarks.landmark:
            landmarks.extend([
                landmark.x,   # x-coordinate
                landmark.y,   # y-coordinate
                landmark.z    # z-coordinate (depth information)
            ])
        
        # Normalize landmarks
        landmarks = np.array(landmarks)
        landmarks = (landmarks - np.min(landmarks)) / (np.max(landmarks) - np.min(landmarks))
        
        return landmarks.reshape(1, -1)

    def advanced_hand_detection(self, frame):
        """
        Enhanced hand detection with multiple visualization and analysis techniques
        """
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Create a copy of the frame for drawing
        output_frame = frame.copy()
        
        # Tracking information
        hand_info = []

        if results.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Advanced Drawing Styles
                self.mp_drawing.draw_landmarks(
                    output_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract and process landmarks
                processed_landmarks = self.extract_enhanced_landmarks(hand_landmarks)
                
                # Predict gesture
                predictions = self.model.predict(processed_landmarks)
                confidence = np.max(predictions)
                pred_index = np.argmax(predictions)
                
                # Safely retrieve gesture information
                gesture_info = self.gesture_classes.get(pred_index, {
                    'name': 'Unknown',
                    'description': 'Unrecognized gesture',
                    'complexity': 'N/A'
                })
                
                # Collect hand tracking information
                hand_info.append({
                    'gesture': gesture_info['name'],
                    'confidence': float(confidence),
                    'description': gesture_info.get('description', 'N/A'),
                    'complexity': gesture_info.get('complexity', 'N/A'),
                    'hand_type': 'Left' if hand_landmarks.landmark[0].x < 0.5 else 'Right'
                })

                # Annotate frame with gesture information
                cv2.putText(output_frame, 
                            f"{gesture_info['name']} ({confidence:.2f})", 
                            (int(hand_landmarks.landmark[0].x * output_frame.shape[1]), 
                             int(hand_landmarks.landmark[0].y * output_frame.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 255, 0), 
                            2)

        return output_frame, hand_info

    def run_hand_detection(self):
        """
        Real-time hand detection and gesture recognition
        """
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame for mirror-like experience
            frame = cv2.flip(frame, 1)

            # Perform advanced hand detection
            processed_frame, hand_detections = self.advanced_hand_detection(frame)

            # Display number of hands detected
            cv2.putText(processed_frame, 
                        f"Hands Detected: {len(hand_detections)}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2)

            # Optional: Print detailed hand information
            for info in hand_detections:
                print(f"Detected: {info['gesture']} (Confidence: {info['confidence']:.2f})")

            cv2.imshow('Comprehensive ASL Gesture Recognition', processed_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    # Initialize the comprehensive recognizer
    recognizer = ComprehensiveASLRecognizer()
    
    # Generate and split synthetic training data
    X = np.random.rand(5000, 63)  # Matches the new landmark extraction shape
    y = np.random.randint(0, len(recognizer.gesture_classes), 5000)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train the model
    print("Training initial model with synthetic data...")
    history = recognizer.model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        verbose=1  # Shows training progress
    )
    
    # Evaluate model performance
    test_loss, test_accuracy = recognizer.model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Run hand detection
    recognizer.run_hand_detection()

if __name__ == "__main__":
    main()