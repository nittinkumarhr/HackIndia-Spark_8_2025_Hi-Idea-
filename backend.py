import cv2
import numpy as np
import mediapipe as mp
import time
import math
from collections import deque
import os
from datetime import datetime
import threading
import re
import uuid

# Optional imports with enhanced error handling
try:
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("Voice feedback not available. Install pyttsx3 for voice features.")

try:
    from onefilter import OneEuroFilter
    ONE_EURO_FILTER_AVAILABLE = True
except ImportError:
    ONE_EURO_FILTER_AVAILABLE = False
    print("OneEuroFilter not available. Install onefilter for enhanced smoothing.")

try:
    import pygame
    from pygame.locals import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("OpenGL visualization not available. Install pygame and PyOpenGL for 3D view.")

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("Speech recognition not available. Install speech_recognition for voice-to-text.")

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from io import BytesIO
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Install matplotlib for math visualization.")

try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("SymPy not available. Install sympy for math equation processing.")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Install tensorflow for enhanced text recognition.")

class SmartAirDrawingSystem:
    def __init__(self):
        # MediaPipe setup with enhanced long-distance detection
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4
        )
        self.keep_original_stroke = True
        # Canvas setup
        self.canvas_width, self.canvas_height = 1280, 720
        self.canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
        
        # 3D space setup
        self.z_min, self.z_max = 0.0, 0.5
        self.z_range = self.z_max - self.z_min
        
        # Drawing properties
        self.line_thickness = 3
        self.letter_colors = {
            'D': (0, 0, 255),
            'R': (0, 255, 0),
            'O': (255, 0, 0),
            'W': (255, 255, 0)
        }
        self.current_letter_index = 0
        self.letter_order = ['D', 'R', 'O', 'W']
        self.current_letter = self.letter_order[self.current_letter_index]
        self.current_color = self.letter_colors[self.current_letter]
        
        # Tracking state
        self.is_drawing = False
        self.last_point = None
        self.gesture_cooldown = 0.5
        self.last_gesture_time = 0
        self.gesture_frame_count = 0
        self.gesture_frame_threshold = 5  # Frames needed to confirm gesture
        self.last_gesture = (0, 0, 0, 0)  # Initialize last_gesture to no fingers up
        
        # Storage for points
        self.current_stroke = []
        self.letter_strokes = {'D': [], 'R': [], 'O': [], 'W': []}
        
        # Initialize advanced smoothing
        self.setup_advanced_smoothing()
        
        # Template paths for letter guidance
        self.letter_templates = {
            'D': self._generate_d_template(),
            'R': self._generate_r_template(),
            'O': self._generate_o_template(),
            'W': self._generate_w_template()
        }
        
        # Visualization settings
        self.show_template = True
        self.show_guide = True
        self.show_3d_view = False
        
        # 3D view settings
        self.rotation_angle = 0
        self.elevation_angle = 30
        self.gl_rotation = [20.0, 30.0, 0.0]
        self.gl_position = [0.0, 0.0, -5.0]
        self.gl_scale = 1.0
        
        # Status tracking
        self.letters_completed = set()
        self.current_message = "Draw letter 'D' in the air"
        self.message_timer = time.time()
        self.message_duration = 3.0
        
        # Mode settings
        self.tutorial_mode = False
        self.challenge_mode = False
        self.tutorial_step = 0
        self.challenge_start_time = 0
        self.challenge_duration = 30.0
        
        # Output directory
        self.output_dir = "air_drawings"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Recognition system
        self.setup_letter_recognition()
        
        # Feedback settings
        self.feedback_active = True
        self.last_recognition_result = None
        self.recognition_confidence = 0.0
        
        # Voice feedback
        self.voice_enabled = False
        if VOICE_AVAILABLE:
            self.setup_voice_feedback()
            
        # Enhanced features
        self.modes = ["draw", "text", "shape", "math", "voice"]
        self.current_mode_index = 0
        self.mode = self.modes[self.current_mode_index]
        self.recognized_text = []
        self.recognized_shapes = []
        self.math_expressions = []
        self.voice_notes = []
        
        # Text replacement
        self.text_buffer = ""
        self.text_position = (50, 50)
        self.text_color = (0, 0, 0)
        self.text_size = 1.0
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Shape recognition
        self.shape_buffer = []
        self.recognized_shape = None
        self.shape_position = (0, 0)
        self.shape_size = (0, 0)
        
        # Math recognition
        self.math_buffer = []
        self.math_expression = ""
        self.math_result = None
        self.math_image = None
        
        # Voice recognition
        self.is_recording = False
        self.voice_thread = None
        self.voice_buffer = ""
        if SPEECH_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
        
        # Split screen
        self.split_screen = True
        self.camera_frame = None
        
        # TensorFlow setup
        self.TF_AVAILABLE = False
        if TF_AVAILABLE:
            try:
                import tensorflow as tf
                self.TF_AVAILABLE = True
                model_path = 'emnist_model.h5'
                if os.path.exists(model_path):
                    self.text_model = tf.keras.models.load_model(model_path)
                    self.text_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                else:
                    print(f"EMNIST model not found at {model_path}. TensorFlow recognition disabled.")
                    self.TF_AVAILABLE = False
            except Exception as e:
                print(f"Failed to load TensorFlow model: {e}. Falling back to template matching.")
                self.TF_AVAILABLE = False

    def setup_advanced_smoothing(self):
        if ONE_EURO_FILTER_AVAILABLE:
            self.position_filters = {
                'x': OneEuroFilter(freq=60, mincutoff=1.0, beta=0.005),
                'y': OneEuroFilter(freq=60, mincutoff=1.0, beta=0.005),
                'z': OneEuroFilter(freq=60, mincutoff=1.0, beta=0.005)
            }
        else:
            self.smoothing_window = 7
            self.position_history = deque(maxlen=self.smoothing_window)

    def setup_letter_recognition(self):
        self.recognition_templates = {}
        for letter in self.letter_order:
            self.recognition_templates[letter] = self._preprocess_template(self.letter_templates[letter])
        
        self.alphabet_templates = {}
        for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if char not in self.letter_templates:
                self.alphabet_templates[char] = self._generate_letter_template(char)
        
        self.shape_templates = {
            'circle': self._generate_circle_template(),
            'rectangle': self._generate_rectangle_template(),
            'triangle': self._generate_triangle_template(),
            'line': self._generate_line_template(),
            'arrow': self._generate_arrow_template(),
            'pentagon': self._generate_pentagon_template(),
            'hexagon': self._generate_hexagon_template(),
            'star': self._generate_star_template(),
            'heart': self._generate_heart_template(),
            'ellipse': self._generate_ellipse_template(),
            'trapezoid': self._generate_trapezoid_template(),
            'diamond': self._generate_diamond_template()
        }

    def setup_voice_feedback(self):
        if VOICE_AVAILABLE:
            try:
                self.voice_engine = pyttsx3.init()
                self.voice_enabled = True
                self.last_voice_time = 0
                self.voice_cooldown = 2.0
                self.voice_engine.setProperty('rate', 150)
                self.voice_engine.setProperty('volume', 0.8)
                self.speak("Smart Air Drawing System ready")
                return True
            except Exception as e:
                print(f"Error initializing voice: {e}")
                self.voice_enabled = False
        return False

    def _generate_letter_template(self, letter):
        template = []
        center_x, center_y = 640, 360
        size = 150
        if letter in self.letter_templates:
            return self.letter_templates[letter]
        for i in range(21):
            t = i / 20.0
            x = int(center_x + size * math.cos(2 * math.pi * t))
            y = int(center_y + size * math.sin(2 * math.pi * t))
            template.append((x, y, 0.25))
        return template

    def _generate_circle_template(self):
        template = []
        center_x, center_y = 640, 360
        radius = 150
        for i in range(21):
            angle = i * 2 * math.pi / 20
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            template.append((x, y, 0.25))
        return template

    def _generate_ellipse_template(self):
        template = []
        center_x, center_y = 640, 360
        a, b = 180, 120
        for i in range(21):
            angle = i * 2 * math.pi / 20
            x = int(center_x + a * math.cos(angle))
            y = int(center_y + b * math.sin(angle))
            template.append((x, y, 0.25))
        return template

    def _generate_trapezoid_template(self):
        template = []
        points = [
            (540, 210, 0.25),
            (740, 210, 0.25),
            (790, 510, 0.25),
            (490, 510, 0.25),
            (540, 210, 0.25)
        ]
        for i in range(len(points)-1):
            p1 = points[i]
            p2 = points[i+1]
            for j in range(11):
                t = j / 10.0
                x = int(p1[0] * (1-t) + p2[0] * t)
                y = int(p1[1] * (1-t) + p2[1] * t)
                template.append((x, y, 0.25))
        return template

    def _generate_diamond_template(self):
        template = []
        points = [
            (640, 210, 0.25),
            (790, 360, 0.25),
            (640, 510, 0.25),
            (490, 360, 0.25),
            (640, 210, 0.25)
        ]
        for i in range(len(points)-1):
            p1 = points[i]
            p2 = points[i+1]
            for j in range(11):
                t = j / 10.0
                x = int(p1[0] * (1-t) + p2[0] * t)
                y = int(p1[1] * (1-t) + p2[1] * t)
                template.append((x, y, 0.25))
        return template

    def _generate_rectangle_template(self):
        template = []
        x1, y1 = 490, 210
        x2, y2 = 790, 510
        for i in range(11):
            t = i / 10.0
            template.append((int(x1 + t * (x2 - x1)), int(y1), 0.25))
            template.append((x2, int(y1 + t * (y2 - y1)), 0.25))
            template.append((int(x2 - t * (x2 - x1)), y2, 0.25))
            template.append((x1, int(y2 - t * (y2 - y1)), 0.25))
        return template

    def _generate_triangle_template(self):
        template = []
        x1, y1 = 640, 210
        x2, y2 = 490, 510
        x3, y3 = 790, 510
        for i in range(11):
            t = i / 10.0
            template.append((int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1)), 0.25))
            template.append((int(x2 + t * (x3 - x2)), int(y2 + t * (y3 - y2)), 0.25))
            template.append((int(x3 + t * (x1 - x3)), int(y3 + t * (y1 - y3)), 0.25))
        return template

    def _generate_line_template(self):
        template = []
        x1, y1 = 490, 360
        x2, y2 = 790, 360
        for i in range(21):
            t = i / 20.0
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            template.append((x, y, 0.25))
        return template

    def _generate_arrow_template(self):
        template = []
        x1, y1 = 490, 360
        x2, y2 = 790, 360
        for i in range(11):
            t = i / 10.0
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            template.append((x, y, 0.25))
        head_length = 75
        head_width = 45
        template.append((int(x2 - head_length), int(y2 - head_width), 0.25))
        template.append((x2, y2, 0.25))
        template.append((int(x2 - head_length), int(y2 + head_width), 0.25))
        template.append((x2, y2, 0.25))
        return template

    def _generate_pentagon_template(self):
        template = []
        center_x, center_y = 640, 360
        radius = 150
        for i in range(5):
            angle = i * 2 * math.pi / 5 - math.pi / 2
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            template.append((x, y, 0.25))
        template.append(template[0])
        return template

    def _generate_hexagon_template(self):
        template = []
        center_x, center_y = 640, 360
        radius = 150
        for i in range(6):
            angle = i * 2 * math.pi / 6 - math.pi / 2
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            template.append((x, y, 0.25))
        template.append(template[0])
        return template

    def _generate_star_template(self):
        template = []
        center_x, center_y = 640, 360
        outer_radius = 150
        inner_radius = 75
        for i in range(10):
            radius = outer_radius if i % 2 == 0 else inner_radius
            angle = i * math.pi / 5 - math.pi / 2
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            template.append((x, y, 0.25))
        template.append(template[0])
        return template

    def _generate_heart_template(self):
        template = []
        center_x, center_y = 640, 360
        for t in np.linspace(0, 2 * math.pi, 21):
            x = int(center_x + 150 * (math.sin(t) ** 3))
            y = int(center_y - 120 * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)) / 16)
            template.append((x, y, 0.25))
        return template

    def _generate_d_template(self):
        template = []
        for i in range(21):
            y = int(150 + i * 22.5)
            template.append((490, y, 0.25))
        for i in range(11):
            angle = i * math.pi / 10
            x = int(490 + 150 * math.sin(angle))
            y = int(360 + 225 * math.cos(angle))
            template.append((x, y, 0.25))
        return template

    def _generate_r_template(self):
        template = []
        for i in range(21):
            y = int(150 + i * 22.5)
            template.append((490, y, 0.25))
        for i in range(11):
            angle = i * math.pi / 10
            x = int(490 + 120 * math.sin(angle))
            y = int(210 + 75 * math.cos(angle))
            template.append((x, y, 0.25))
        for i in range(11):
            x = int(490 + i * 12)
            y = int(285 + i * 22.5)
            template.append((x, y, 0.25))
        return template

    def _generate_o_template(self):
        template = []
        for i in range(21):
            angle = i * 2 * math.pi / 20
            x = int(640 + 150 * math.cos(angle))
            y = int(360 + 150 * math.sin(angle))
            template.append((x, y, 0.25))
        return template

    def _generate_w_template(self):
        template = []
        points = [
            (540, 150, 0.25),
            (590, 450, 0.25),
            (640, 250, 0.25),
            (690, 450, 0.25),
            (740, 150, 0.25)
        ]
        for i in range(len(points)-1):
            p1 = points[i]
            p2 = points[i+1]
            for j in range(11):
                t = j / 10.0
                x = int(p1[0] * (1-t) + p2[0] * t)
                y = int(p1[1] * (1-t) + p2[1] * t)
                template.append((x, y, 0.25))
        return template

    def _preprocess_template(self, points):
        points_2d = [(p[0], p[1]) for p in points]
        if len(points_2d) > 64:
            indices = np.linspace(0, len(points_2d) - 1, 64).astype(int)
            resampled = [points_2d[i] for i in indices]
        else:
            resampled = points_2d
        x_vals = [p[0] for p in resampled]
        y_vals = [p[1] for p in resampled]
        centroid_x = sum(x_vals) / len(resampled) if resampled else 0
        centroid_y = sum(y_vals) / len(resampled) if resampled else 0
        max_dist = max([math.sqrt((x - centroid_x)**2 + (y - centroid_y)**2) 
                        for x, y in zip(x_vals, y_vals)] + [1e-6])
        normalized = [((x - centroid_x) / max_dist, (y - centroid_y) / max_dist) 
                    for x, y in zip(x_vals, y_vals)]
        return normalized

    def finger_is_up(self, hand_landmarks, finger_tip_id, finger_dip_id):
        """Check if a finger is extended by comparing tip and DIP joint positions."""
        return hand_landmarks.landmark[finger_tip_id].y < hand_landmarks.landmark[finger_dip_id].y

    def detect_finger_gesture(self, hand_landmarks):
        """Detect finger-based gestures and return the action to perform."""
        # Get finger states
        index_up = self.finger_is_up(hand_landmarks, 8, 6)   # Index finger
        middle_up = self.finger_is_up(hand_landmarks, 12, 10) # Middle finger
        ring_up = self.finger_is_up(hand_landmarks, 16, 14)   # Ring finger
        pinky_up = self.finger_is_up(hand_landmarks, 20, 18)  # Pinky finger

        # Create gesture tuple
        gesture = (int(index_up), int(middle_up), int(ring_up), int(pinky_up))

        # Map gestures to actions
        gesture_actions = {
            (1, 0, 0, 0): self.perform_task,  # Index only: Perform current mode task
            (0, 1, 0, 0): self.switch_mode,   # Middle only: Switch mode
            (1, 1, 1, 1): self.clear_canvas   # All four fingers: Clear canvas
        }

        return gesture, gesture_actions.get(gesture, None)

    def perform_task(self, x, y, z):
        """Perform the task associated with the current mode."""
        if self.mode == "draw" or self.mode == "text" or self.mode == "shape" or self.mode == "math":
            if not self.is_drawing:
                self.is_drawing = True
                self.last_point = (x, y, z)
                self.current_stroke = [(x, y, z)]
            else:
                if self.last_point and self._is_valid_point(x, y):
                    depth_factor = 1.0 - z
                    thickness = max(1, int(self.line_thickness * depth_factor))
                    cv2.line(
                        self.canvas,
                        (self.last_point[0], self.last_point[1]),
                        (x, y),
                        self.current_color,
                        thickness
                    )
                self.last_point = (x, y, z)
                self.current_stroke.append((x, y, z))
        elif self.mode == "voice":
            self.start_voice_recognition()

    def switch_mode(self):
        """Switch to the next mode in the modes list."""
        self.current_mode_index = (self.current_mode_index + 1) % len(self.modes)
        self.mode = self.modes[self.current_mode_index]
        self.current_message = f"Mode changed to: {self.mode.upper()}"
        self.message_timer = time.time()
        if self.voice_enabled:
            self.speak(f"Mode changed to {self.mode}")

    def smooth_position(self, position):
        if ONE_EURO_FILTER_AVAILABLE:
            filtered_x = self.position_filters['x'](position[0])
            filtered_y = self.position_filters['y'](position[1])
            filtered_z = self.position_filters['z'](position[2])
            return (int(filtered_x), int(filtered_y), filtered_z)
        else:
            self.position_history.append(position)
            if len(self.position_history) < 2:
                return position
            total_weight = 0
            smooth_x, smooth_y, smooth_z = 0, 0, 0
            for i, pos in enumerate(self.position_history):
                weight = i + 1
                total_weight += weight
                smooth_x += pos[0] * weight
                smooth_y += pos[1] * weight
                smooth_z += pos[2] * weight
            return (
                int(smooth_x / total_weight),
                int(smooth_y / total_weight),
                smooth_z / total_weight
            )

    def compute_hu_moments(self, points):
        if len(points) < 3:
            return None
        points_array = np.array(points, dtype=np.float32)
        moments = cv2.moments(points_array)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        return hu_moments

    def recognize_stroke(self, stroke):
        if len(stroke) < 10:
            return None, 0.0
        
        stroke_2d = [(p[0], p[1]) for p in stroke]
        if len(stroke_2d) > 64:
            indices = np.linspace(0, len(stroke_2d) - 1, 64).astype(int)
            resampled = [stroke_2d[i] for i in indices]
        else:
            resampled = stroke_2d
        
        stroke_hu = self.compute_hu_moments(resampled)
        if stroke_hu is None:
            return None, 0.0
        
        x_vals = [p[0] for p in resampled]
        y_vals = [p[1] for p in resampled]
        centroid_x = sum(x_vals) / len(resampled)
        centroid_y = sum(y_vals) / len(resampled)
        max_dist = max([math.sqrt((x - centroid_x)**2 + (y - centroid_y)**2) 
                    for x, y in zip(x_vals, y_vals)] + [1e-6])
        normalized = [((x - centroid_x) / max_dist, (y - centroid_y) / max_dist) 
                    for x, y in zip(x_vals, y_vals)]
        
        if self.mode == "draw" or self.mode == "text":
            best_score = 0
            best_letter = None
            if getattr(self, 'TF_AVAILABLE', False) and self.mode == "text" and hasattr(self, 'text_model'):
                try:
                    stroke_img = self.stroke_to_image(stroke)
                    if stroke_img is not None:
                        pred = self.text_model.predict(stroke_img)
                        best_idx = np.argmax(pred[0])
                        best_score = pred[0][best_idx]
                        best_letter = self.text_labels[best_idx]
                        if best_score > 0.7:
                            return best_letter, best_score
                except Exception as e:
                    print(f"TensorFlow prediction failed: {e}")
            all_templates = {**self.recognition_templates, **self.alphabet_templates}
            for letter, template in all_templates.items():
                if len(template) == 0 or len(normalized) == 0:
                    continue
                total_distance = 0
                for i, point in enumerate(normalized):
                    template_idx = min(i, len(template) - 1)
                    template_point = template[template_idx]
                    distance = math.sqrt((point[0] - template_point[0])**2 + 
                                    (point[1] - template_point[1])**2)
                    total_distance += distance
                avg_distance = total_distance / len(normalized)
                score = 1.0 - min(avg_distance, 1.0)
                if score > best_score:
                    best_score = score
                    best_letter = letter
            return best_letter, best_score
        
        elif self.mode == "shape":
            best_score = 0
            best_shape = None
            template_hu_moments = {}
            for shape, template in self.shape_templates.items():
                template_points = [(p[0], p[1]) for p in template]
                template_hu = self.compute_hu_moments(template_points)
                if template_hu is not None:
                    template_hu_moments[shape] = template_hu
            
            points_array = np.array(stroke_2d, dtype=np.int32)
            hull = cv2.convexHull(points_array)
            epsilon = 0.03 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            num_vertices = len(approx)
            
            for shape, template in self.shape_templates.items():
                template_normalized = self._preprocess_template(template)
                if len(template_normalized) == 0 or len(normalized) == 0:
                    continue
                total_distance = 0
                for i, point in enumerate(normalized):
                    template_idx = min(i, len(template_normalized) - 1)
                    template_point = template_normalized[template_idx]
                    distance = math.sqrt((point[0] - template_point[0])**2 + 
                                    (point[1] - template_point[1])**2)
                    total_distance += distance
                avg_distance = total_distance / len(normalized)
                template_score = 1.0 - min(avg_distance, 1.0)
                
                hu_score = 0
                if shape in template_hu_moments and stroke_hu is not None:
                    hu_diff = np.abs(stroke_hu - template_hu_moments[shape])
                    hu_score = 1.0 / (1.0 + np.mean(hu_diff))
                
                combined_score = 0.6 * template_score + 0.4 * hu_score
                
                if shape == "circle" and self.is_circle(stroke_2d):
                    combined_score = max(combined_score, 0.85)
                elif shape == "rectangle" and self.is_rectangle(stroke_2d):
                    combined_score = max(combined_score, 0.85)
                elif shape == "triangle" and self.is_triangle(stroke_2d):
                    combined_score = max(combined_score, 0.85)
                elif shape == "pentagon" and num_vertices == 5:
                    combined_score = max(combined_score, 0.80)
                elif shape == "hexagon" and num_vertices == 6:
                    combined_score = max(combined_score, 0.80)
                elif shape == "star" and self.is_star(stroke_2d):
                    combined_score = max(combined_score, 0.75)
                elif shape == "heart" and self.is_heart(stroke_2d):
                    combined_score = max(combined_score, 0.75)
                elif shape == "ellipse" and self.is_ellipse(stroke_2d):
                    combined_score = max(combined_score, 0.80)
                elif shape == "trapezoid" and self.is_trapezoid(stroke_2d):
                    combined_score = max(combined_score, 0.80)
                elif shape == "diamond" and num_vertices == 4:
                    combined_score = max(combined_score, 0.80)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_shape = shape
            return best_shape, best_score
        
        elif self.mode == "math":
            math_symbols = {}
            symbol_methods = {
                'plus': self._generate_plus_template,
                'minus': self._generate_minus_template,
                'times': self._generate_times_template,
                'divide': self._generate_divide_template,
                'equals': self._generate_equals_template,
                'x': self._generate_x_template,
                'y': self._generate_y_template,
                'sin': self._generate_sin_template,
                'cos': self._generate_cos_template,
                'sqrt': self._generate_sqrt_template,
                '(': lambda: self._generate_parenthesis_template(True),
                ')': lambda: self._generate_parenthesis_template(False)
            }
            for symbol, method in symbol_methods.items():
                try:
                    math_symbols[symbol] = method()
                except Exception as e:
                    print(f"Warning: Template for {symbol} not available: {e}")
                    continue
            for i in range(10):
                try:
                    math_symbols[str(i)] = self._generate_number_template(i)
                except Exception as e:
                    print(f"Warning: Template for number {i} not available: {e}")
                    continue
            
            best_score = 0
            best_symbol = None
            stroke_hu = self.compute_hu_moments([(p[0], p[1]) for p in stroke])
            for symbol, template in math_symbols.items():
                if not template:
                    continue
                template_normalized = self._preprocess_template(template)
                if len(template_normalized) == 0 or len(normalized) == 0:
                    continue
                total_distance = 0
                for i, point in enumerate(normalized):
                    template_idx = min(i, len(template_normalized) - 1)
                    template_point = template_normalized[template_idx]
                    distance = math.sqrt((point[0] - template_point[0])**2 + 
                                    (point[1] - template_point[1])**2)
                    total_distance += distance
                avg_distance = total_distance / len(normalized)
                template_score = 1.0 - min(avg_distance, 1.0)
                
                hu_score = 0
                if stroke_hu is not None:
                    template_hu = self.compute_hu_moments([(p[0], p[1]) for p in template])
                    if template_hu is not None:
                        hu_diff = np.abs(stroke_hu - template_hu)
                        hu_score = 1.0 / (1.0 + np.mean(hu_diff))
                
                combined_score = 0.6 * template_score + 0.4 * hu_score
                if combined_score > best_score:
                    best_score = combined_score
                    best_symbol = symbol
            return best_symbol, best_score
        
        return None, 0.0

    def _generate_sqrt_template(self):
        template = []
        points = [
            (590, 360),
            (620, 360),
            (620, 510),
            (740, 210),
            (790, 210)
        ]
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            for j in range(11):
                t = j / 10.0
                x = int(p1[0] * (1 - t) + p2[0] * t)
                y = int(p1[1] * (1 - t) + p2[1] * t)
                template.append((x, y, 0.25))
        return template

    def segment_stroke(self, stroke):
        if len(stroke) < 20:
            return [stroke]
        segments = []
        current_segment = [stroke[0]]
        x_vals = [p[0] for p in stroke]
        x_min, x_max = min(x_vals), max(x_vals)
        x_range = x_max - x_min
        threshold = x_range / 4 if x_range > 0 else 10
        for i in range(1, len(stroke)):
            prev_x = stroke[i-1][0]
            curr_x = stroke[i][0]
            if abs(curr_x - prev_x) > threshold and len(current_segment) > 10:
                segments.append(current_segment)
                current_segment = [stroke[i]]
            else:
                current_segment.append(stroke[i])
        if len(current_segment) > 10:
            segments.append(current_segment)
        return segments

    def _generate_parenthesis_template(self, is_open=True):
        template = []
        center_x, center_y = 640, 360
        height = 150
        for i in range(21):
            t = i / 20.0
            angle = math.pi * (t - 0.5)
            x_offset = 50 * math.cos(angle) * (-1 if is_open else 1)
            x = int(center_x + x_offset)
            y = int(center_y + height * math.sin(angle))
            template.append((x, y, 0.25))
        return template

    def _generate_number_template(self, number):
        templates = {
            '0': [(640, 210), (790, 210), (790, 510), (490, 510), (490, 210), (640, 210)],
            '1': [(640, 210), (640, 510)],
            '2': [(490, 210), (790, 210), (790, 360), (490, 360), (490, 510), (790, 510)],
            '3': [(490, 210), (790, 210), (790, 360), (490, 360), (790, 360), (790, 510), (490, 510)],
            '4': [(490, 360), (790, 360), (640, 360), (640, 210), (640, 510)],
            '5': [(790, 210), (490, 210), (490, 360), (790, 360), (790, 510), (490, 510)],
            '6': [(790, 210), (490, 210), (490, 510), (790, 510), (790, 360), (490, 360)],
            '7': [(490, 210), (790, 210), (490, 510)],
            '8': [(490, 210), (790, 210), (790, 510), (490, 510), (490, 210), (790, 360), (490, 360)],
            '9': [(490, 510), (790, 510), (790, 210), (490, 210), (490, 360), (790, 360)]
        }
        points = templates.get(str(number), [])
        template = []
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            for j in range(11):
                t = j / 10.0
                x = int(p1[0] * (1 - t) + p2[0] * t)
                y = int(p1[1] * (1 - t) + p2[1] * t)
                template.append((x, y, 0.25))
        return template

    def is_ellipse(self, points):
        if len(points) < 10:
            return False
        points_array = np.array(points, dtype=np.int32)
        if len(points_array) < 5:
            return False
        ellipse = cv2.fitEllipse(points_array)
        (center, axes, angle) = ellipse
        major_axis, minor_axis = max(axes), min(axes)
        if minor_axis == 0:
            return False
        aspect_ratio = major_axis / minor_axis
        return 1.2 <= aspect_ratio <= 3.0

    def is_trapezoid(self, points):
        if len(points) < 8:
            return False
        points_array = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points_array)
        epsilon = 0.03 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) != 4:
            return False
        top = approx[0][0] - approx[1][0]
        bottom = approx[2][0] - approx[3][0]
        top_slope = top[1] / top[0] if top[0] != 0 else float('inf')
        bottom_slope = bottom[1] / bottom[0] if bottom[0] != 0 else float('inf')
        return abs(top_slope - bottom_slope) < 0.1

    def is_diamond(self, points):
        return self.is_polygon(points, 4)

    def _generate_plus_template(self):
        template = []
        for i in range(11):
            x = int(590 + i * 15)
            template.append((x, 360, 0.25))
        for i in range(11):
            y = int(285 + i * 15)
            template.append((640, y, 0.25))
        return template

    def _generate_minus_template(self):
        template = []
        for i in range(11):
            x = int(590 + i * 15)
            template.append((x, 360, 0.25))
        return template

    def _generate_times_template(self):
        template = []
        for i in range(11):
            t = i / 10.0
            x = int(590 + t * 150)
            y = int(285 + t * 150)
            template.append((x, y, 0.25))
        for i in range(11):
            t = i / 10.0
            x = int(740 - t * 150)
            y = int(285 + t * 150)
            template.append((x, y, 0.25))
        return template

    def _generate_divide_template(self):
        template = []
        for i in range(11):
            x = int(590 + i * 15)
            template.append((x, 360, 0.25))
        template.append((640, 330, 0.25))
        template.append((640, 390, 0.25))
        return template

    def _generate_equals_template(self):
        template = []
        for i in range(11):
            x = int(590 + i * 15)
            template.append((x, 340, 0.25))
        for i in range(11):
            x = int(590 + i * 15)
            template.append((x, 380, 0.25))
        return template

    def _generate_x_template(self):
        template = []
        for i in range(11):
            t = i / 10.0
            x = int(590 + t * 150)
            y = int(285 + t * 150)
            template.append((x, y, 0.25))
        for i in range(11):
            t = i / 10.0
            x = int(740 - t * 150)
            y = int(285 + t * 150)
            template.append((x, y, 0.25))
        return template

    def _generate_y_template(self):
        template = []
        for i in range(6):
            t = i / 5.0
            x = int(640 - t * 75)
            y = int(285 + t * 75)
            template.append((x, y, 0.25))
        for i in range(6):
            t = i / 5.0
            x = int(640 + t * 75)
            y = int(285 + t * 75)
            template.append((x, y, 0.25))
        for i in range(6):
            y = int(360 + i * 15)
            template.append((640, y, 0.25))
        return template

    def _generate_sin_template(self):
        template = []
        for i in range(21):
            x = int(590 + i * 7.5)
            y = int(360 + 75 * math.sin(i * 2 * math.pi / 20))
            template.append((x, y, 0.25))
        return template

    def _generate_cos_template(self):
        template = []
        for i in range(21):
            x = int(590 + i * 7.5)
            y = int(360 + 75 * math.cos(i * 2 * math.pi / 20))
            template.append((x, y, 0.25))
        return template

    def is_circle(self, points):
        if len(points) < 10:
            return False
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        center_x = sum(x_vals) / len(points)
        center_y = sum(y_vals) / len(points)
        radii = [math.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in points]
        avg_radius = sum(radii) / len(radii)
        std_radius = math.sqrt(sum((r - avg_radius)**2 for r in radii) / len(radii))
        return std_radius / avg_radius < 0.15 and len(points) > 20

    def is_rectangle(self, points):
        if len(points) < 8:
            return False
        points_array = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points_array)
        epsilon = 0.03 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4:
            angles = []
            for i in range(4):
                p1 = approx[i][0]
                p2 = approx[(i+1) % 4][0]
                p3 = approx[(i+2) % 4][0]
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                if mag1 * mag2 == 0:
                    continue
                cos_angle = dot / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))
                angle = math.acos(cos_angle) * 180 / math.pi
                angles.append(angle)
            return all(abs(angle - 90) < 15 for angle in angles)
        return False

    def is_triangle(self, points):
        if len(points) < 6:
            return False
        points_array = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points_array)
        epsilon = 0.03 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        return len(approx) == 3

    def is_polygon(self, points, sides):
        if len(points) < sides:
            return False
        points_array = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points_array)
        epsilon = 0.03 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        return len(approx) == sides

    def is_star(self, points):
        if len(points) < 10:
            return False
        points_array = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points_array, returnPoints=False)
        if len(hull) < 3:
            return False
        defects = cv2.convexityDefects(points_array, hull)
        return defects is not None and len(defects) >= 5

    def is_heart(self, points):
        if len(points) < 10:
            return False
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        min_y = min(y_vals)
        max_y = max(y_vals)
        mid_y = (min_y + max_y) / 2
        top_half = [(x, y) for x, y in points if y < mid_y]
        bottom_half = [(x, y) for x, y in points if y >= mid_y]
        if len(top_half) < 4 or len(bottom_half) < 4:
            return False
        top_x_vals = [x for x, _ in top_half]
        if len(top_x_vals) < 2:
            return False
        dip_x = (min(top_x_vals) + max(top_x_vals)) / 2
        bottom_points = [(x, y) for x, y in bottom_half if abs(x - dip_x) < (max(top_x_vals) - min(top_x_vals)) / 4]
        return len(bottom_points) > 2 and max([y for _, y in bottom_points]) > mid_y

    def is_convexity_defects(self, points):
        points_arraykick = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points_array, returnPoints=False)
        if len(hull) < 3:
            return False
        defects = cv2.convexityDefects(points_array, hull)
        return defects is not None and len(defects) >= 5

    def clear_canvas(self):
        self.canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
        self.current_stroke = []
        self.recognized_text = []
        self.recognized_shapes = []
        self.math_expressions = []
        self.voice_notes = []
        self.current_message = "Canvas cleared"
        self.message_timer = time.time()
        if self.voice_enabled:
            self.speak("Canvas cleared")

    def save_drawing(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/air_drawing_{timestamp}.png"
        cv2.imwrite(filename, self.canvas)
        self.current_message = f"Drawing saved as {filename}"
        self.message_timer = time.time()
        if self.voice_enabled:
            self.speak("Drawing saved")
        return filename

    def speak(self, text):
        if not self.voice_enabled:
            return
        current_time = time.time()
        if current_time - self.last_voice_time < self.voice_cooldown:
            return
        try:
            self.voice_engine.say(text)
            self.voice_engine.runAndWait()
            self.last_voice_time = current_time
        except Exception as e:
            print(f"Voice feedback error: {e}")

    def start_voice_recognition(self):
        if not SPEECH_AVAILABLE:
            self.current_message = "Speech recognition not available"
            self.message_timer = time.time()
            return
        if self.is_recording:
            return
        self.is_recording = True
        self.current_message = "Listening..."
        self.message_timer = time.time()
        if self.voice_enabled:
            self.speak("Listening")
        self.voice_thread = threading.Thread(target=self._record_audio)
        self.voice_thread.daemon = True
        self.voice_thread.start()

    def _record_audio(self):
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=5)
            try:
                text = self.recognizer.recognize_google(audio)
                self.voice_buffer = text
                if "draw something" in text.lower():
                    self.mode = "draw"
                    self.current_mode_index = self.modes.index("draw")
                    self.clear_canvas()
                    self.current_message = "Ready to draw! Start creating on a blank canvas."
                    self.message_timer = time.time()
                    if self.voice_enabled:
                        self.speak("Ready to draw! Start creating.")
                elif self.mode == "shape":
                    shape = self._parse_shape_from_text(text)
                    if shape:
                        self.recognized_shapes.append((shape, self.text_position, None))
                        self.draw_perfect_shape(shape, [(self.text_position[0], self.text_position[1])])
                        self.current_message = f"Recognized shape: {shape}"
                        self.message_timer = time.time()
                        if self.voice_enabled:
                            self.speak(f"Shape {shape} drawn")
                elif self.mode == "math":
                    math_expr = self._parse_math_from_text(text)
                    if math_expr:
                        math_img = self.process_math_expression(math_expr)
                        if math_img is not None:
                            h, w = math_img.shape[:2]
                            roi = self.canvas[self.text_position[1]:self.text_position[1]+h, 
                                            self.text_position[0]:self.text_position[0]+w]
                            if roi.shape[0] >= h and roi.shape[1] >= w:
                                mask = cv2.cvtColor(math_img, cv2.COLOR_BGR2GRAY)
                                _, mask = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY_INV)
                                roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
                                roi_fg = cv2.bitwise_and(math_img, math_img, mask=mask)
                                self.canvas[self.text_position[1]:self.text_position[1]+h, 
                                          self.text_position[0]:self.text_position[0]+w] = cv2.add(roi_bg, roi_fg)
                            self.math_expressions.append((math_expr, self.text_position))
                            self.text_position = (self.text_position[0], self.text_position[1] + h + 10)
                            self.current_message = f"Recognized math: {math_expr}"
                            self.message_timer = time.time()
                            if self.voice_enabled:
                                self.speak("Math expression added")
                else:
                    self.voice_notes.append((text, self.text_position))
                    self.text_position = (self.text_position[0], self.text_position[1] + 40)
                    self.current_message = f"Recognized: {text}"
                    self.message_timer = time.time()
                    if self.voice_enabled:
                        self.speak("Voice note added")
            except sr.UnknownValueError:
                self.current_message = "Could not understand audio"
                self.message_timer = time.time()
            except sr.RequestError as e:
                self.current_message = f"Error: {e}"
                self.message_timer = time.time()
        except Exception as e:
            self.current_message = f"Error: {e}"
            self.message_timer = time.time()
        finally:
            self.is_recording = False

    def _parse_shape_from_text(self, text):
        text = text.lower()
        shapes = {
            "circle": "circle",
            "rectangle": "rectangle",
            "square": "rectangle",
            "triangle": "triangle",
            "line": "line",
            "arrow": "arrow",
            "pentagon": "pentagon",
            "hexagon": "hexagon",
            "star": "star",
            "heart": "heart"
        }
        for key, shape in shapes.items():
            if key in text:
                return shape
        return None

    def _parse_math_from_text(self, text):
        text = text.lower()
        text = re.sub(r'\bplus\b', '+', text)
        text = re.sub(r'\bminus\b', '-', text)
        text = re.sub(r'\btimes\b|\bx\b', '*', text)
        text = re.sub(r'\bdivided by\b|\bdivide\b', '/', text)
        text = re.sub(r'\bequals\b', '=', text)
        text = re.sub(r'\bsquared\b', '^2', text)
        text = re.sub(r'\bcubed\b', '^3', text)
        text = re.sub(r'\bsine\b', 'sin', text)
        text = re.sub(r'\bcosine\b', 'cos', text)
        return text.strip()

    def process_math_expression(self, expression_str):
        if not MATPLOTLIB_AVAILABLE or not SYMPY_AVAILABLE:
            return None
        try:
            expr = sympy.sympify(expression_str, evaluate=False)
            if '=' in expression_str:
                var = sympy.Symbol('x')
                eq = sympy.Eq(expr.lhs, expr.rhs)
                solutions = sympy.solve(eq, var)
                result_text = f"Solutions: {solutions}"
            elif any(func in str(expr) for func in ['sin', 'cos', 'tan']):
                x = np.linspace(-10, 10, 100)
                y = [float(expr.subs('x', val)) for val in x]
                fig = plt.figure(figsize=(6, 4))
                plt.plot(x, y, 'b-', label=str(expr))
                plt.grid(True)
                plt.legend()
                plt.title(f"Plot of ${sympy.latex(expr)}$")
            else:
                result = expr.evalf()
                result_text = f"${sympy.latex(expr)} = {sympy.latex(result)}$"
                fig = plt.figure(figsize=(4, 1))
                plt.text(0.5, 0.5, result_text, fontsize=14, ha='center', va='center')
                plt.axis('off')
            
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = BytesIO()
            canvas.print_png(buf)
            buf.seek(0)
            img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()
            img = cv2.imdecode(img, 1)
            plt.close(fig)
            return img
        except Exception as e:
            print(f"Math processing error: {e}")
            return None

    def draw_perfect_shape(self, shape_type, points):
        if len(points) < 2:
            return
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        min_x, max_x = min(x_vals), max(x_vals)
        min_y, max_y = min(y_vals), max(y_vals)
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        width = max_x - min_x
        height = max_y - min_y
        radius = max(width, height) // 2
        
        if shape_type == "circle":
            cv2.circle(self.canvas, (center_x, center_y), radius, (0, 0, 255), 2)
            self.recognized_shapes.append(("circle", (center_x, center_y), radius))
        elif shape_type == "rectangle":
            cv2.rectangle(self.canvas, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            self.recognized_shapes.append(("rectangle", (min_x, min_y), (max_x, max_y)))
        elif shape_type == "triangle":
            height = int(math.sqrt(3) * width / 2)
            p1 = (center_x, min_y)
            p2 = (min_x, min_y + height)
            p3 = (max_x, min_y + height)
            cv2.line(self.canvas, p1, p2, (255, 0, 0), 2)
            cv2.line(self.canvas, p2, p3, (255, 0, 0), 2)
            cv2.line(self.canvas, p3, p1, (255, 0, 0), 2)
            self.recognized_shapes.append(("triangle", (p1, p2, p3)))
        elif shape_type == "line":
            cv2.line(self.canvas, (min_x, center_y), (max_x, center_y), (0, 0, 0), 2)
            self.recognized_shapes.append(("line", (min_x, center_y), (max_x, center_y)))
        elif shape_type == "arrow":
            cv2.line(self.canvas, (min_x, center_y), (max_x, center_y), (0, 0, 0), 2)
            head_length = width // 5
            head_width = height // 3
            cv2.line(self.canvas, (max_x, center_y), 
                    (max_x - head_length, center_y - head_width), (0, 0, 0), 2)
            cv2.line(self.canvas, (max_x, center_y), 
                    (max_x - head_length, center_y + head_width), (0, 0, 0), 2)
            self.recognized_shapes.append(("arrow", (min_x, center_y), (max_x, center_y)))
        elif shape_type == "pentagon":
            points = []
            for i in range(5):
                angle = i * 2 * math.pi / 5 - math.pi / 2
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                points.append([int(x), int(y)])
            cv2.polylines(self.canvas, [np.array(points)], True, (128, 0, 128), 2)
            self.recognized_shapes.append(("pentagon", points))
        elif shape_type == "hexagon":
            points = []
            for i in range(6):
                angle = i * 2 * math.pi / 6 - math.pi / 2
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                points.append([int(x), int(y)])
            cv2.polylines(self.canvas, [np.array(points)], True, (0, 128, 128), 2)
            self.recognized_shapes.append(("hexagon", points))
        elif shape_type == "star":
            points = []
            for i in range(10):
                radius = radius if i % 2 == 0 else radius // 2
                angle = i * math.pi / 5 - math.pi / 2
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                points.append([int(x), int(y)])
            cv2.polylines(self.canvas, [np.array(points)], True, (255, 255, 0), 2)
            self.recognized_shapes.append(("star", points))
        elif shape_type == "heart":
            points = []
            for t in np.linspace(0, 2 * math.pi, 21):
                x = center_x + radius * (math.sin(t) ** 3)
                y = center_y - radius * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)) / 16
                points.append([int(x), int(y)])
            cv2.polylines(self.canvas, [np.array(points)], True, (255, 0, 255), 2)
            self.recognized_shapes.append(("heart", points))
        elif shape_type == "ellipse":
            axes = (width // 2, height // 2)
            cv2.ellipse(self.canvas, (center_x, center_y), axes, 0, 0, 360, (0, 255, 255), 2)
            self.recognized_shapes.append(("ellipse", (center_x, center_y), axes))
        elif shape_type == "trapezoid":
            points = [
                [center_x - width // 4, min_y],
                [center_x + width // 4, min_y],
                [center_x + width // 2, max_y],
                [center_x - width // 2, max_y]
            ]
            cv2.polylines(self.canvas, [np.array(points)], True, (255, 128, 0), 2)
            self.recognized_shapes.append(("trapezoid", points))
        elif shape_type == "diamond":
            points = [
                [center_x, min_y],
                [max_x, center_y],
                [center_x, max_y],
                [min_x, center_y]
            ]
            cv2.polylines(self.canvas, [np.array(points)], True, (128, 128, 255), 2)
            self.recognized_shapes.append(("diamond", points))

    def preprocess_frame(self, frame):
        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        return frame

    def process_frame(self, frame):
        self.camera_frame = frame.copy()
        display_frame = self.preprocess_frame(frame)
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        display_canvas = self.canvas.copy()

        if self.show_template and self.mode == "draw":
            self._draw_template(display_canvas)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                index_tip = hand_landmarks.landmark[8]
                x = int(index_tip.x * self.canvas_width)
                y = int(index_tip.y * self.canvas_height)
                z = (index_tip.z - self.z_min) / self.z_range
                z = max(0.0, min(1.0, z))
                smooth_pos = self.smooth_position((x, y, z))
                x, y, z = smooth_pos

                cursor_size = max(3, int(15 * (1.0 - z)))
                cursor_color = (
                    int(self.current_color[0] * (1.0 - z / 2)),
                    int(self.current_color[1] * (1.0 - z / 2)),
                    int(self.current_color[2] * (1.0 - z / 2))
                )
                cv2.circle(display_frame, (x, y), cursor_size, cursor_color, -1)

                current_time = time.time()
                if current_time - self.last_gesture_time < self.gesture_cooldown:
                    continue

                gesture, action = self.detect_finger_gesture(hand_landmarks)
                if gesture == self.last_gesture:
                    self.gesture_frame_count += 1
                else:
                    self.gesture_frame_count = 1
                    self.last_gesture = gesture

                if self.gesture_frame_count >= self.gesture_frame_threshold and action is not None:
                    if action == self.perform_task:
                        action(x, y, z)
                    else:
                        action()
                        self.last_gesture_time = current_time
                        self.gesture_frame_count = 0

                if not (gesture == (1, 0, 0, 0)) and self.is_drawing:
                    self.is_drawing = False
                    self.last_point = None
                    self._process_completed_stroke(display_canvas)

        if self.mode == "math" and self.math_buffer:
            expr_preview = " ".join(self.math_buffer)
            try:
                sympy.sympify(expr_preview, evaluate=False)
                preview_color = (0, 255, 0)
            except:
                preview_color = (0, 0, 255)
            cv2.putText(
                display_canvas,
                f"Expression: {expr_preview}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                preview_color,
                2
            )

        self._add_ui_elements(display_canvas)

        if self.challenge_mode:
            self._update_challenge(display_canvas)

        return display_frame, display_canvas

    def _is_valid_point(self, x, y):
        return 0 <= x < self.canvas_width and 0 <= y < self.canvas_height

    def _set_feedback(self, message, speak=True):
        self.current_message = message
        self.message_timer = time.time()
        if self.voice_enabled and speak:
            self.speak(message)

    def _process_completed_stroke(self, display_canvas):
        if not self.current_stroke or len(self.current_stroke) < 10:
            self.current_stroke = []
            return

        try:
            if self.mode == "draw":
                self.letter_strokes[self.current_letter].append(self.current_stroke)
                if self.feedback_active:
                    result = self.recognize_stroke(self.current_stroke)
                    if result is not None:
                        letter, confidence = result
                        self.last_recognition_result = letter
                        self.recognition_confidence = confidence
                        if confidence > 0.6 and letter is not None:
                            if letter == self.current_letter:
                                self._set_feedback(f"Good job! That looks like {letter}")
                            else:
                                self._set_feedback(f"That looks like {letter}, not {self.current_letter}")

            elif self.mode == "text":
                characters = self.segment_stroke(self.current_stroke)
                for char_stroke in characters:
                    if len(char_stroke) > 10:
                        result = self.recognize_stroke(char_stroke)
                        if result is not None:
                            letter, confidence = result
                            if confidence > 0.5 and letter:
                                self.text_buffer += letter
                                x_vals = [p[0] for p in char_stroke]
                                y_vals = [p[1] for p in char_stroke]
                                min_x, max_x = min(x_vals), max(x_vals)
                                min_y, max_y = min(y_vals), max(y_vals)
                                stroke_height = max_y - min_y
                                desired_height = max(30, min(100, stroke_height))
                                base_font_scale = 1.0
                                text_size = cv2.getTextSize(
                                    letter, self.text_font, base_font_scale, 2
                                )[0]
                                font_scale = desired_height / text_size[1] if text_size[1] > 0 else 1.0
                                text_pos = (min_x, max_y)
                                cv2.putText(
                                    self.canvas,
                                    letter,
                                    text_pos,
                                    self.text_font,
                                    font_scale,
                                    self.text_color,
                                    2
                                )
                                self.recognized_text.append((letter, text_pos))
                                text_width = cv2.getTextSize(
                                    letter, self.text_font, font_scale, 2
                                )[0][0]
                                self.text_position = (text_pos[0] + text_width + 10, text_pos[1])
                                self._set_feedback(f"Recognized: {letter}")

            elif self.mode == "shape":
                result = self.recognize_stroke(self.current_stroke)
                if result is not None:
                    shape, confidence = result
                    if confidence > 0.5 and shape:
                        if not self.keep_original_stroke:
                            self.canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
                            for prev_shape, pos, size in self.recognized_shapes:
                                self.draw_perfect_shape(prev_shape, [pos] if isinstance(pos, tuple) else pos)
                        points = [(int(p[0]), int(p[1])) for p in self.current_stroke]
                        for i in range(1, len(points)):
                            cv2.line(
                                self.canvas,
                                points[i-1],
                                points[i],
                                (128, 128, 128),
                                1,
                                lineType=cv2.LINE_AA
                            )
                        self.draw_perfect_shape(shape, [(p[0], p[1]) for p in self.current_stroke])
                        self._set_feedback(f"Recognized shape: {shape} (Toggle original stroke: 'k')")

            elif self.mode == "math":
                result = self.recognize_stroke(self.current_stroke)
                if result is not None:
                    symbol, confidence = result
                    if confidence > 0.5 and symbol:
                        self.math_buffer.append(symbol)
                        math_expr = " ".join(self.math_buffer)
                        math_img = self.process_math_expression(math_expr)
                        if math_img is not None:
                            h, w = math_img.shape[:2]
                            roi = self.canvas[self.text_position[1]:self.text_position[1] + h,
                                            self.text_position[0]:self.text_position[0] + w]
                            if roi.shape[0] >= h and roi.shape[1] >= w:
                                mask = cv2.cvtColor(math_img, cv2.COLOR_BGR2GRAY)
                                _, mask = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY_INV)
                                roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
                                roi_fg = cv2.bitwise_and(math_img, math_img, mask=mask)
                                self.canvas[self.text_position[1]:self.text_position[1] + h,
                                        self.text_position[0]:self.text_position[0] + w] = cv2.add(roi_bg, roi_fg)
                            self.math_expressions.append((math_expr, self.text_position))
                            self.text_position = (self.text_position[0], self.text_position[1] + h + 10)
                            self._set_feedback(f"Math symbol: {symbol}")
        except Exception as e:
            print(f"Error processing stroke: {e}")
            self._set_feedback(f"Error: {str(e)}", speak=False)

        self.current_stroke = []

    def _draw_template(self, canvas):
            if not self.show_template:
                return
            template = self.letter_templates.get(self.current_letter, [])
            for i in range(len(template) - 1):
                p1 = (int(template[i][0]), int(template[i][1]))
                p2 = (int(template[i + 1][0]), int(template[i + 1][1]))
                cv2.line(canvas, p1, p2, (200, 200, 200), 1, lineType=cv2.LINE_AA)

    def _add_ui_elements(self, canvas):
        cv2.putText(
            canvas,
            f"Mode: {self.mode.upper()}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        cv2.putText(
            canvas,
            f"Letter: {self.current_letter}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.current_color,
            2
        )
        if time.time() - self.message_timer < self.message_duration:
            cv2.putText(
                canvas,
                self.current_message,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )
        if self.tutorial_mode:
            cv2.putText(
                canvas,
                f"Tutorial Step {self.tutorial_step + 1}: {self.current_message}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        instructions = [
            "Index finger: Draw/Perform task",
            "Middle finger: Switch mode",
            "All four fingers: Clear canvas",
            "Keys: T(Template), G(Guide), S(Save), C(Challenge), V(Voice), K(Keep Stroke)"
        ]
        for i, instruction in enumerate(instructions):
            cv2.putText(
                canvas,
                instruction,
                (20, self.canvas_height - 100 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

    def _update_challenge(self, canvas):
        elapsed = time.time() - self.challenge_start_time
        remaining = self.challenge_duration - elapsed
        if remaining <= 0:
            self.challenge_mode = False
            completed = len(self.letters_completed)
            self._set_feedback(f"Challenge Over! Completed: {completed}/4")
            return
        cv2.putText(
            canvas,
            f"Time: {remaining:.1f}s",
            (self.canvas_width - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    def stroke_to_image(self, stroke):
        if len(stroke) < 5:
            return None
        stroke_canvas = np.zeros((self.canvas_height, self.canvas_width), dtype=np.uint8)
        points = [(int(p[0]), int(p[1])) for p in stroke]
        for i in range(1, len(points)):
            cv2.line(stroke_canvas, points[i-1], points[i], 255, 3, lineType=cv2.LINE_AA)
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        min_x, max_x = min(x_vals), max(x_vals)
        min_y, max_y = min(y_vals), max(y_vals)
        margin = 20
        min_x, max_x = max(0, min_x - margin), min(self.canvas_width, max_x + margin)
        min_y, max_y = max(0, min_y - margin), min(self.canvas_height, max_y + margin)
        if max_x <= min_x or max_y <= min_y:
            return None
        cropped = stroke_canvas[min_y:max_y, min_x:max_x]
        resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
        resized = resized.reshape(1, 28, 28, 1) / 255.0
        return resized

    def toggle_template(self):
        self.show_template = not self.show_template
        self._set_feedback(f"Template {'on' if self.show_template else 'off'}")

    def toggle_guide(self):
        self.show_guide = not self.show_guide
        self._set_feedback(f"Guide {'on' if self.show_guide else 'off'}")

    def toggle_challenge(self):
        self.challenge_mode = not self.challenge_mode
        if self.challenge_mode:
            self.challenge_start_time = time.time()
            self.letters_completed = set()
            self._set_feedback("Challenge Mode: Draw all letters in 30s!")
        else:
            self._set_feedback("Challenge Mode Off")

    def toggle_voice(self):
        if SPEECH_AVAILABLE:
            self.start_voice_recognition()
        else:
            self._set_feedback("Speech recognition not available")

    def toggle_keep_stroke(self):
        self.keep_original_stroke = not self.keep_original_stroke
        self._set_feedback(f"Keep original stroke: {'on' if self.keep_original_stroke else 'off'}")

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.canvas_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.canvas_height)

        if OPENGL_AVAILABLE and self.show_3d_view:
            pygame.init()
            pygame.display.set_mode((self.canvas_width // 2, self.canvas_height // 2), DOUBLEBUF | OPENGL)
            gluPerspective(45, (self.canvas_width / self.canvas_height), 0.1, 50.0)
            glTranslatef(0.0, 0.0, -5)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Resize frame to match canvas dimensions and ensure 3 channels
            frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))
            if len(frame.shape) == 2:  # Convert grayscale to BGR if necessary
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            display_frame, display_canvas = self.process_frame(frame)

            # Ensure display_frame matches the expected shape
            display_frame = cv2.resize(display_frame, (self.canvas_width, self.canvas_height))
            if len(display_frame.shape) == 2:
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

            if self.split_screen:
                split_frame = np.zeros((self.canvas_height, self.canvas_width * 2, 3), dtype=np.uint8)
                split_frame[:, :self.canvas_width] = display_frame
                split_frame[:, self.canvas_width:] = display_canvas
                cv2.imshow("Smart Air Drawing", split_frame)
            else:
                cv2.imshow("Smart Air Drawing", display_canvas)

            if OPENGL_AVAILABLE and self.show_3d_view:
                self._render_3d_view()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                self.toggle_template()
            elif key == ord('g'):
                self.toggle_guide()
            elif key == ord('s'):
                self.save_drawing()
            elif key == ord('c'):
                self.toggle_challenge()
            elif key == ord('v'):
                self.toggle_voice()
            elif key == ord('k'):
                self.toggle_keep_stroke()

        cap.release()
        cv2.destroyAllWindows()
        if OPENGL_AVAILABLE and self.show_3d_view:
            pygame.quit()

    def _render_3d_view(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluPerspective(45, (self.canvas_width / self.canvas_height), 0.1, 50.0)
        glTranslatef(self.gl_position[0], self.gl_position[1], self.gl_position[2])
        glRotatef(self.gl_rotation[0], 1, 0, 0)
        glRotatef(self.gl_rotation[1], 0, 1, 0)
        glRotatef(self.gl_rotation[2], 0, 0, 1)
        glScalef(self.gl_scale, self.gl_scale, self.gl_scale)

        glBegin(GL_LINES)
        for stroke in self.letter_strokes[self.current_letter]:
            for i in range(len(stroke) - 1):
                p1 = stroke[i]
                p2 = stroke[i + 1]
                x1 = (p1[0] / self.canvas_width) * 2 - 1
                y1 = -((p1[1] / self.canvas_height) * 2 - 1)
                z1 = p1[2]
                x2 = (p2[0] / self.canvas_width) * 2 - 1
                y2 = -((p2[1] / self.canvas_height) * 2 - 1)
                z2 = p2[2]
                color = [c / 255.0 for c in self.current_color]
                glColor3f(color[0], color[1], color[2])
                glVertex3f(x1, y1, z1)
                glVertex3f(x2, y2, z2)
            glEnd()

        pygame.display.flip()

if __name__ == "__main__":
    system = SmartAirDrawingSystem()
    system.run()