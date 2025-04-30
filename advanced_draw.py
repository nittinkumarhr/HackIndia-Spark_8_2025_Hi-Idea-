import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import time
import random
import math
import collections
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

class SignLanguageVisualizer:
    def __init__(self, width=640, height=480):
      # Adjusted resolution for web app
        self.frame_width = width
        self.frame_height = height
        self.sequence_length = 30  # For LSTM - make sure this line is executed

        # MediaPipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.landmark_style = self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3)
        self.connection_style = self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )

        self.gesture_classes = {
            0: {'name': 'A', 'description': 'Letter A in ASL'},
            1: {'name': 'B', 'description': 'Letter B in ASL'},
            2: {'name': 'C', 'description': 'Letter C in ASL'},
            3: {'name': 'L', 'description': 'Letter L in ASL'},
            4: {'name': 'O', 'description': 'Letter O in ASL'},
            5: {'name': 'Y', 'description': 'Letter Y in ASL'}
        }
        self.num_classes = len(self.gesture_classes)
        self.model = self.create_model()

        # Drawing Setup
        self.canvas_color = (255, 255, 255)  # White canvas
        self.canvas = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * self.canvas_color[0]
        self.drawing_color = (0, 0, 0)
        self.eraser_color = self.canvas_color
        self.base_thickness = 5
        self.eraser_thickness = 20
        self.max_thickness_variation = 3
        self.jitter_amount = 1  # Reduced jitter for smoother lines
        self.speed_influence = 0.1
        self.max_draw_speed = 50
        self.smoothing_length = 10  # Increased for better smoothing

        # Drawing State
        self.prev_point = None
        self.point_history = collections.deque(maxlen=self.smoothing_length)
        self.velocity_history = collections.deque(maxlen=self.smoothing_length)
        self.is_drawing = False
        self.is_erasing = False
        self.last_button_activation = 0
        self.button_cooldown = 15
        self.button_feedback_frames = 0
        self.feedback_duration = 5
        self.frame_count = 0
        self.last_point_time = time.time()

        # Button Setup
        self.buttons = {}
        self.button_height = 50
        self.button_margin = 15
        self.colors = {
            "Black": (0, 0, 0),
            "Red": (0, 0, 255),
            "Blue": (255, 0, 0),
            "Green": (0, 255, 0),
            "Purple": (255, 0, 255),
            "Yellow": (0, 255, 255),
        }
        self.setup_buttons()
        
        self.hover_button = None
        self.active_button = None
        
        # Full screen state
        self.is_fullscreen = False
        self.fullscreen_target = "canvas"  # Which view is in fullscreen: "canvas" or "camera"

        # Draw a line to separate drawing area from buttons
        self.draw_separator_line()

    def draw_separator_line(self):
        # Draw a line to separate the buttons area from the drawing area
        separator_y = self.button_height + self.button_margin * 2
        cv2.line(self.canvas, 
                (0, separator_y), 
                (self.frame_width, separator_y), 
                (200, 200, 200), 2)  # Light gray line

    def setup_buttons(self):
        button_y = self.button_margin
        
        # Position the color buttons on the left side
        button_width_color = 70
        button_x_color = self.button_margin
        
        # Add color buttons
        for name, color_val in self.colors.items():
            self.buttons[name] = {'rect': (button_x_color, button_y, button_x_color + button_width_color, button_y + self.button_height), 
                                 'color': color_val}  # Light gray line
            button_x_color += button_width_color + 5  # Smaller margin between color buttons
        
        # Add Eraser button after color buttons
        self.buttons["Eraser"] = {'rect': (button_x_color, button_y, button_x_color + 100, button_y + self.button_height), 
                                'color': (200, 200, 200)}
        
        # Remove Clear button from the canvas UI

    def create_model(self):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, 63)),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def extract_landmarks(self, hand_landmarks):
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        landmarks_np = np.array(landmarks)
        wrist = landmarks_np[0:3]
        landmarks_relative = []
        min_val, max_val = float('inf'), float('-inf')
        for i in range(0, len(landmarks_np), 3):
            lm_point = landmarks_np[i:i+3] - wrist
            landmarks_relative.extend(lm_point)
            min_val = min(min_val, *lm_point)
            max_val = max(max_val, *lm_point)
        scale = max_val - min_val or 1e-6
        landmarks_normalized = (np.array(landmarks_relative) - min_val) / scale
        return landmarks_normalized

    def draw_buttons(self, frame, is_fullscreen=False, fullscreen_target=None):
        # Skip drawing buttons in fullscreen camera mode
        if is_fullscreen and fullscreen_target == "camera":
            return
            
        for name, button_info in self.buttons.items():
            x1, y1, x2, y2 = button_info['rect']
            base_color = button_info['color']
            border_color = (0, 0, 0)

            if name == self.active_button and self.button_feedback_frames > 0:
                color = tuple(int(c * 0.5) % 256 for c in base_color)
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), color, -1)
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), border_color, 2)
            elif name == self.hover_button:
                color = tuple(int(c * 1.2) % 256 for c in base_color)
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), color, -1)
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), border_color, 2)
            else:
                color = base_color
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1)

            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y1 + (y2 - y1 + text_size[1]) // 2
            text_color = (255, 255, 255) if np.mean(color) < 128 else (0, 0, 0)
            if name == "Eraser": text_color = (0, 0, 0)
            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

        # Draw a separator line to clearly separate buttons from drawing area
        # Skip drawing separator in fullscreen camera mode
        if not (is_fullscreen and fullscreen_target == "camera"):
            separator_y = self.button_height + self.button_margin * 2
            cv2.line(frame, (0, separator_y), (self.frame_width, separator_y), (100, 100, 100), 2)

        if self.button_feedback_frames > 0:
            self.button_feedback_frames -= 1
            if self.button_feedback_frames == 0:
                self.active_button = None

    def clear_canvas(self):
        """Clear the canvas and reset drawing state"""
        self.canvas = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * self.canvas_color[0]
        self.prev_point = None
        self.point_history.clear()
        self.velocity_history.clear()
        # Redraw the separator line
        self.draw_separator_line()

    def adaptive_smoothing(self, current_point):
        """Apply adaptive smoothing based on velocity and acceleration"""
        now = time.time()
        dt = now - self.last_point_time
        self.last_point_time = now
        
        # Add current point to history
        self.point_history.append(current_point)
        
        if len(self.point_history) < 2:
            return current_point
            
        # Calculate velocity
        if dt > 0:
            if len(self.point_history) >= 2:
                p1 = np.array(self.point_history[-1])
                p2 = np.array(self.point_history[-2])
                velocity = np.linalg.norm(p1 - p2) / max(dt, 0.001)
                self.velocity_history.append(velocity)
        
        # Adaptive weighting based on velocity
        if len(self.velocity_history) > 0:
            avg_velocity = np.mean(self.velocity_history)
            # For faster movements, use more aggressive smoothing
            smoothing_factor = min(0.9, 0.3 + 0.5 * (avg_velocity / 100))
        else:
            smoothing_factor = 0.5
            
        # Apply cubic spline or weighted average for smoothing
        if len(self.point_history) >= 3:
            # More weight to recent points, less to older ones
            weights = np.linspace(0.1, 1.0, len(self.point_history)) ** 2
            weights = weights / np.sum(weights)
            smooth_point = np.average(self.point_history, axis=0, weights=weights).astype(int)
            
            # Add small amount of prediction for lag compensation
            if len(self.point_history) >= 4 and len(self.velocity_history) >= 2:
                # Calculate prediction vector based on recent movement
                pred_vector = np.array(self.point_history[-1]) - np.array(self.point_history[-4])
                # Scale down the prediction
                pred_scale = 0.1
                prediction = np.array(smooth_point) + pred_vector * pred_scale
                # Blend the prediction with smoothed point
                smooth_point = (smooth_point * (1 - smoothing_factor) + prediction * smoothing_factor).astype(int)
                
            return tuple(smooth_point)
        else:
            return current_point

    def process_frame(self, frame, is_fullscreen=False, fullscreen_target=None):
        self.frame_count += 1
        
        # Resize frame to match the current resolution settings
        if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        # Create copies for different purposes
        output_frame = frame.copy()
        
        # Ensure canvas matches frame dimensions
        if self.canvas.shape[:2] != (self.frame_height, self.frame_width):
            new_canvas = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * self.canvas_color[0]
            # Copy old canvas content if possible
            h, w = min(self.canvas.shape[0], new_canvas.shape[0]), min(self.canvas.shape[1], new_canvas.shape[1])
            new_canvas[:h, :w] = self.canvas[:h, :w]
            self.canvas = new_canvas
            # Redraw the separator line
            self.draw_separator_line()

        # Create a clean canvas copy for fullscreen display (without drawing UI elements on it)
        clean_canvas = self.canvas.copy()
            
        # Draw buttons only on output_frame when appropriate
        if not (is_fullscreen and fullscreen_target == "camera"):
            # Draw buttons on the output frame (not directly on canvas)
            self.draw_buttons(output_frame, is_fullscreen, fullscreen_target)

        # Skip hand processing in fullscreen canvas mode (only show the clean canvas)
        if is_fullscreen and fullscreen_target == "canvas":
            return output_frame, clean_canvas

        current_point = None
        index_up = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks only on the output frame (not on canvas)
                self.mp_drawing.draw_landmarks(output_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                              self.landmark_style, self.connection_style)

                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
                current_point = (int(index_tip.x * self.frame_width), int(index_tip.y * self.frame_height))

                index_up = index_tip.y < index_pip.y - 0.02
                
                # Show drawing cursor with appropriate size only on output_frame (not canvas)
                cursor_size = self.eraser_thickness if self.is_erasing else self.base_thickness
                cursor_color = self.drawing_color if not self.is_erasing else (0, 0, 255)
                cv2.circle(output_frame, current_point, cursor_size, cursor_color, -1 if index_up else 2)

                if index_up and self.frame_count - self.last_button_activation > self.button_cooldown:
                    for name, button_info in self.buttons.items():
                        x1, y1, x2, y2 = button_info['rect']
                        if x1 < current_point[0] < x2 and y1 < current_point[1] < y2:
                            self.hover_button = name
                            self.last_button_activation = self.frame_count
                            self.active_button = name
                            self.button_feedback_frames = self.feedback_duration
                            if name == "Eraser":
                                self.drawing_color = self.eraser_color
                                self.is_erasing = True
                            elif name in self.colors:
                                self.drawing_color = self.colors[name]
                                self.is_erasing = False
                            break
                    else:
                        self.hover_button = None

        # Get the separator line y-coordinate
        separator_y = self.button_height + self.button_margin * 2

        if index_up and current_point and self.frame_count - self.last_button_activation > self.button_cooldown:
            # Only draw if below the separator line
            if current_point[1] > separator_y:
                # Add subtle jitter for natural hand-drawn look
                draw_point = (current_point[0] + random.randint(-self.jitter_amount, self.jitter_amount),
                              current_point[1] + random.randint(-self.jitter_amount, self.jitter_amount))
                
                # Apply advanced smoothing
                smooth_point = self.adaptive_smoothing(draw_point)

                if self.is_drawing and self.prev_point is not None:
                    distance = math.dist(self.prev_point, smooth_point)
                    
                    # Dynamic thickness based on drawing speed
                    speed_factor = max(0, 1 - (distance / self.max_draw_speed) * self.speed_influence)
                    thickness = (self.eraser_thickness if self.is_erasing else self.base_thickness) + int(self.max_thickness_variation * speed_factor)
                    thickness = max(1, min(thickness, (self.eraser_thickness if self.is_erasing else self.base_thickness) + self.max_thickness_variation))
                    
                    # For more natural drawing, use Bézier curve when drawing quickly
                    if distance > 20 and not self.is_erasing:
                        # Use a simple quadratic Bézier curve for smoother lines
                        mid_point = ((self.prev_point[0] + smooth_point[0]) // 2, (self.prev_point[1] + smooth_point[1]) // 2)
                        # Draw first half of the curve
                        pts = np.array([self.prev_point, mid_point], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(self.canvas, [pts], False, self.drawing_color, thickness, cv2.LINE_AA)
                        # Draw second half of the curve
                        pts = np.array([mid_point, smooth_point], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(self.canvas, [pts], False, self.drawing_color, thickness, cv2.LINE_AA)
                    else:
                        # For slower movements or eraser, use straight lines
                        cv2.line(self.canvas, tuple(self.prev_point), tuple(smooth_point), self.drawing_color, thickness, cv2.LINE_AA)
                    
                    # Update the clean canvas copy
                    clean_canvas = self.canvas.copy()

                self.is_drawing = True
                self.prev_point = smooth_point
            else:
                # If above the separator line, don't draw
                self.is_drawing = False
                self.prev_point = None
        else:
            self.is_drawing = False
            self.prev_point = None
            self.point_history.clear()
            self.velocity_history.clear()

        # In fullscreen mode, return appropriate frames based on the target
        if is_fullscreen:
            if fullscreen_target == "canvas":
                # Return clean canvas without UI elements when in canvas fullscreen
                return output_frame, clean_canvas
            else:  # Camera fullscreen
                # Return camera feed without canvas elements
                return output_frame, self.canvas.copy()
        else:
            # In regular mode, return both frames with UI elements
            return output_frame, self.canvas.copy()

def main():
    st.set_page_config(
        page_title="Sign Language Visualizer",
        layout="wide"
    )
    
    st.title("Sign Language Visualizer Web App")
    
    # App description and instructions
    with st.expander("Instructions", expanded=False):
        st.markdown("""
        ## How to Use
        1. Allow camera access when prompted
        2. Show your hand to the camera
        3. Point with your index finger to draw
        4. Use the buttons at the top to change colors and use the eraser
        5. Use the sidebar fullscreen options to view camera or canvas in fullscreen
        6. Clear the canvas using the Clear button in the sidebar
        7. Draw only below the horizontal separator line
        """)
    
    # Session state for persistent settings
    if 'resolution' not in st.session_state:
        st.session_state.resolution = "640x480"
    if 'visualizer' not in st.session_state:
        width, height = map(int, st.session_state.resolution.split('x'))
        st.session_state.visualizer = SignLanguageVisualizer(width=width, height=height)
    if 'is_fullscreen' not in st.session_state:
        st.session_state.is_fullscreen = False
    if 'fullscreen_target' not in st.session_state:
        st.session_state.fullscreen_target = "canvas"
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Resolution selection with apply button to prevent constant reinitialization
    new_resolution = st.sidebar.selectbox(
        "Select Resolution",
        ["640x480", "800x600", "1280x720"],
        index=["640x480", "800x600", "1280x720"].index(st.session_state.resolution)
    )
    
    # Apply resolution change button
    if st.sidebar.button("Apply Resolution"):
        if new_resolution != st.session_state.resolution:
            st.session_state.resolution = new_resolution
            width, height = map(int, new_resolution.split('x'))
            
            # Create new visualizer with updated resolution
            old_canvas = st.session_state.visualizer.canvas
            st.session_state.visualizer = SignLanguageVisualizer(width=width, height=height)
            
            # Try to preserve the existing drawing by resizing it
            st.session_state.visualizer.canvas = cv2.resize(old_canvas, (width, height))
            
            # Draw the separator line
            st.session_state.visualizer.draw_separator_line()
            
            # Use st.rerun() instead of deprecated experimental_rerun
            st.rerun()
    
    # Add Clear Canvas button to sidebar
    if st.sidebar.button("Clear Canvas"):
        st.session_state.visualizer.clear_canvas()
    
    # Drawing thickness control
    thickness = st.sidebar.slider("Brush Thickness", 1, 15, st.session_state.visualizer.base_thickness)
    if thickness != st.session_state.visualizer.base_thickness:
        st.session_state.visualizer.base_thickness = thickness
    
    # Eraser thickness control
    eraser_thickness = st.sidebar.slider("Eraser Size", 10, 50, st.session_state.visualizer.eraser_thickness)
    if eraser_thickness != st.session_state.visualizer.eraser_thickness:
        st.session_state.visualizer.eraser_thickness = eraser_thickness
    
    # Fullscreen options in sidebar
    st.sidebar.header("Fullscreen Options")
    
    fullscreen_col1, fullscreen_col2 = st.sidebar.columns(2)
    
    with fullscreen_col1:
        if st.button("🖼️ Canvas Fullscreen"):
            st.session_state.is_fullscreen = True
            st.session_state.fullscreen_target = "canvas"
            st.session_state.visualizer.is_fullscreen = True
            st.session_state.visualizer.fullscreen_target = "canvas"
            st.rerun()
    
    with fullscreen_col2:
        if st.button("📹 Camera Fullscreen"):
            st.session_state.is_fullscreen = True
            st.session_state.fullscreen_target = "camera"
            st.session_state.visualizer.is_fullscreen = True
            st.session_state.visualizer.fullscreen_target = "camera"
            st.rerun()
    
    if st.sidebar.button("Exit Fullscreen"):
        st.session_state.is_fullscreen = False
        st.session_state.visualizer.is_fullscreen = False
        st.rerun()
    
    # Save canvas button
    if st.sidebar.button("Save Drawing"):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"drawing_{timestamp}.png"
        cv2.imwrite(filename, st.session_state.visualizer.canvas)
        st.sidebar.success(f"Drawing saved as {filename}")
        
        # Offer download link
        with open(filename, "rb") as file:
            btn = st.sidebar.download_button(
                label="Download Drawing",
                data=file,
                file_name=filename,
                mime="image/png"
            )
    
    # Setup columns for camera and canvas
    if not st.session_state.is_fullscreen:
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Camera Feed")
            camera_placeholder = st.empty()
        
        with col2:
            st.header("Drawing Canvas")
            canvas_placeholder = st.empty()
        
        # Full screen container (empty in regular mode)
        fullscreen_placeholder = st.empty()
    else:
        # Hide regular columns in fullscreen mode
        col1 = col2 = None
        camera_placeholder = canvas_placeholder = None
        fullscreen_placeholder = st.empty()
    
    # Status indicator
    status = st.empty()
    
    status.info("Starting camera feed...")
    
    # Video capture from webcam
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    width, height = map(int, st.session_state.resolution.split('x'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your camera connection.")
        st.stop()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from camera. Please check your connection.")
                break
            
            # Flip horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process the frame - pass fullscreen information to process_frame
            output_frame, canvas = st.session_state.visualizer.process_frame(
                frame, 
                is_fullscreen=st.session_state.is_fullscreen,
                fullscreen_target=st.session_state.fullscreen_target
            )
            
            # Convert to RGB for display
            output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            
            # Handle fullscreen mode
            if st.session_state.is_fullscreen:
                # Show ONLY the selected view in fullscreen
                if st.session_state.fullscreen_target == "canvas":
                    # In canvas fullscreen, show only the canvas without buttons
                    fullscreen_placeholder.image(canvas_rgb, channels="RGB", use_container_width=True)
                    status.success("Fullscreen canvas mode. Use the sidebar to exit fullscreen.")
                else:
                    # In camera fullscreen, show only the camera feed
                    fullscreen_placeholder.image(output_frame_rgb, channels="RGB", use_container_width=True)
                    status.success("Fullscreen camera mode. Use the sidebar to exit fullscreen.")
            else:
                # Display frames in regular columns
                if col1 and col2 and camera_placeholder and canvas_placeholder:
                    camera_placeholder.image(output_frame_rgb, channels="RGB", use_container_width=True)
                    canvas_placeholder.image(canvas_rgb, channels="RGB", use_container_width=True)
                    status.success("Camera feed active. Point with your index finger to draw below the line.")
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.01)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        cap.release()
        status.warning("Camera feed stopped.")

if __name__ == "__main__":
    main()
    
    