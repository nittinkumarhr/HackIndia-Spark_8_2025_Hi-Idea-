import streamlit as st
import numpy as np
import cv2
from collections import deque
import time
from scipy.interpolate import splprep, splev
import base64
from io import BytesIO
from PIL import Image
import threading
import google.generativeai as genai
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from dotenv import load_dotenv
import importlib.util
import sys
import subprocess

class VirtualPaintWebApp:
    def __init__(self):
        self.bpoints = [deque(maxlen=1024)]
        self.gpoints = [deque(maxlen=1024)]
        self.rpoints = [deque(maxlen=1024)]
        self.ypoints = [deque(maxlen=1024)]
        
        self.blue_index = 0
        self.green_index = 0
        self.red_index = 0
        self.yellow_index = 0

        self.kernel = np.ones((5, 5), np.uint8)
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        self.color_names = ["BLUE", "GREEN", "RED", "YELLOW"]
        self.colorIndex = 0
        
        self.paintWindow = np.zeros((1200, 1200, 3), dtype=np.uint8) + 255 
        self.setupCanvas()
        
        self.upper_hsv = np.array([153, 255, 255])
        self.lower_hsv = np.array([64, 72, 49])
        
        self.ai_ask_mode = False
        self.analysis_running = False
        self.image_path = "screen_shot/Nitintemp.png"

    def setupCanvas(self):
        """Sets up the drawing canvas with color selection buttons."""
        cv2.rectangle(self.paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)  # CLEAR
        cv2.rectangle(self.paintWindow, (200, 1), (350, 65), self.colors[0], 2)  # BLUE
        cv2.rectangle(self.paintWindow, (400, 1), (550, 65), self.colors[1], 2)  # GREEN
        cv2.rectangle(self.paintWindow, (600, 1), (750, 65), self.colors[2], 2)  # RED
        cv2.rectangle(self.paintWindow, (800, 1), (950, 65), self.colors[3], 2)  # YELLOW
        
        cv2.putText(self.paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(self.paintWindow, "BLUE", (245, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[0], 2, cv2.LINE_AA)
        cv2.putText(self.paintWindow, "GREEN", (445, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[1], 2, cv2.LINE_AA)
        cv2.putText(self.paintWindow, "RED", (645, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[2], 2, cv2.LINE_AA)
        cv2.putText(self.paintWindow, "YELLOW", (845, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[3], 2, cv2.LINE_AA)

    def clearCanvas(self):
        self.bpoints = [deque(maxlen=1024)]
        self.gpoints = [deque(maxlen=1024)]
        self.rpoints = [deque(maxlen=1024)]
        self.ypoints = [deque(maxlen=1024)]
        self.blue_index = self.green_index = self.red_index = self.yellow_index = 0
        self.paintWindow[:, :, :] = 255
        self.setupCanvas()

    def addPoint(self, center):
        if self.colorIndex == 0:
            if center is None:
                self.bpoints.append(deque(maxlen=1024))
                self.blue_index += 1
            else:
                if len(self.bpoints) <= self.blue_index:
                    self.bpoints.append(deque(maxlen=1024))
                self.bpoints[self.blue_index].appendleft(center)
        elif self.colorIndex == 1:
            if center is None:
                self.gpoints.append(deque(maxlen=1024))
                self.green_index += 1
            else:
                if len(self.gpoints) <= self.green_index:
                    self.gpoints.append(deque(maxlen=1024))
                self.gpoints[self.green_index].appendleft(center)
        elif self.colorIndex == 2:
            if center is None:
                self.rpoints.append(deque(maxlen=1024))
                self.red_index += 1
            else:
                if len(self.rpoints) <= self.red_index:
                    self.rpoints.append(deque(maxlen=1024))
                self.rpoints[self.red_index].appendleft(center)
        elif self.colorIndex == 3:
            if center is None:
                self.ypoints.append(deque(maxlen=1024))
                self.yellow_index += 1
            else:
                if len(self.ypoints) <= self.yellow_index:
                    self.ypoints.append(deque(maxlen=1024))
                self.ypoints[self.yellow_index].appendleft(center)

    def interpolate_points(self, points, num_points=50):
        if len(points) < 5:
            return points
        pts = np.array(points, dtype=np.float32)
        if np.any(np.isnan(pts)) or pts.size == 0:
            return points
        try:
            tck, _ = splprep([pts[:, 0], pts[:, 1]], k=3, s=0)
            u_fine = np.linspace(0, 1, num_points)
            x_fine, y_fine = splev(u_fine, tck)
            return list(zip(np.int32(x_fine), np.int32(y_fine)))
        except Exception as e:
            print(f"Error in interpolation: {e}")
            return points

    def drawPointsAsText(self):
        points = [self.bpoints, self.gpoints, self.rpoints, self.ypoints]
        text_chars = list(string.printable)
        for i, color_points in enumerate(points):
            for deque_points in color_points:
                if len(deque_points) > 2:
                    smoothed_points = self.interpolate_points(deque_points)
                    for point in smoothed_points:
                        cv2.putText(self.paintWindow, text_chars[i], (point[0], point[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.2, self.colors[i], 2, cv2.LINE_AA)

    def drawPoints(self):
        self.drawPointsAsText()
        points = [self.bpoints, self.gpoints, self.rpoints, self.ypoints]
        for i, color_points in enumerate(points):
            for deque_points in color_points:
                if len(deque_points) > 2:
                    smoothed_points = self.interpolate_points(deque_points)
                    for point in smoothed_points:
                        cv2.circle(self.paintWindow, point, 3, self.colors[i], -1)

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1200, 1200))  # Increased to 1200x1200

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.erode(mask, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=1)

        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if cnts:
            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            else:
                center = None

            # Adjusted button detection for 1200x1200 (scaled from 600x600)
            if center and center[1] <= 130:  # Scaled from 65 (65 * 2)
                if 80 <= center[0] <= 280:    # CLEAR (40-140 * 2)
                    self.clearCanvas()
                elif 400 <= center[0] <= 700: # BLUE (200-350 * 2)
                    self.colorIndex = 0
                elif 800 <= center[0] <= 1100: # GREEN (400-550 * 2)
                    self.colorIndex = 1
                elif 1200 <= center[0] <= 1500: # RED (600-750 * 2)
                    self.colorIndex = 2
                elif 1600 <= center[0] <= 1900: # YELLOW (800-950 * 2)
                    self.colorIndex = 3
            else:
                if not self.ai_ask_mode and center:
                    self.addPoint(center)
        else:
            if not self.ai_ask_mode:
                self.addPoint(None)

        self.drawPoints()
        cv2.putText(frame, f"Selected Color: {self.color_names[self.colorIndex]}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)  # Increased font size

        return frame

    def save_screenshot(self):
        cropped_paint_window = self.paintWindow[70:, :, :].copy()
        cv2.imwrite(self.image_path, cropped_paint_window)
        return self.image_path

    def generate_analysis(self, api_key):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            self.save_screenshot()
            uploaded_file = genai.upload_file(self.image_path)
            prompt = (
                "Please solve the following drawing image, which may be related to mathematics, "
                "coding, or data structures and algorithms (DSA). "
                "Ensure the solution is beginner-friendly, with a focus on clarity and thorough detailed explanations."
            )
            result = model.generate_content([uploaded_file, prompt])
            if os.path.exists(self.image_path):
                os.remove(self.image_path)
            return result.text if result and hasattr(result, 'text') else "Analysis failed. Please try again."
        except Exception as e:
            return f"Error during analysis: {str(e)}"

class WebcamProcessor(VideoProcessorBase):
    def __init__(self, app):
        self.app = app

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_frame = self.app.process_frame(img)
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

def launch_advanced_draw(canvas_path=None):
    try:
        if canvas_path and 'app' in st.session_state:
            cv2.imwrite(canvas_path, st.session_state.app.paintWindow)
        advanced_draw_path = st.session_state.get('advanced_draw_path', 'advanced_draw.py')
        if not os.path.exists(advanced_draw_path):
            st.error(f"Advanced draw script not found at: {advanced_draw_path}")
            return False
        cmd = [sys.executable, "-m", "streamlit", "run", advanced_draw_path]
        if canvas_path:
            cmd.extend(["--", "--canvas_path", canvas_path])
        process = subprocess.Popen(cmd)
        st.session_state.advanced_draw_process = process
        return True
    except Exception as e:
        st.error(f"Error launching advanced draw: {str(e)}")
        return False

def run_streamlit_app(return_callback=None):
    if return_callback:
        if st.button("← Back to Main Menu"):
            return_callback()
            return
    
    if 'app' not in st.session_state:
        st.session_state.app = VirtualPaintWebApp()
        st.session_state.webcam_started = False
        st.session_state.analysis_result = None
        st.session_state.fullscreen_mode = False
        st.session_state.webcam_placeholder = None
        st.session_state.canvas_placeholder = None
    
    if not st.session_state.fullscreen_mode:
        st.title("Virtual Paint Web App")
        st.markdown("""
        Draw by moving a colored object in front of your webcam. The app will track the object and draw on the canvas.
        Adjust the HSV values to detect your colored object properly.
        """)
    
    if not st.session_state.fullscreen_mode:
        with st.sidebar:
            st.header("Controls")
            color_options = ["BLUE", "GREEN", "RED", "YELLOW"]
            selected_color = st.selectbox("Select Color", color_options, 
                                          index=st.session_state.app.colorIndex)
            st.session_state.app.colorIndex = color_options.index(selected_color)
            if st.button("Clear Canvas"):
                st.session_state.app.clearCanvas()
            st.header("Color Detection Settings")
            st.subheader("Color Range - Lower Bounds")
            lower_h = st.slider("Color Type (Lower)", 0, 180, int(st.session_state.app.lower_hsv[0]), 
                           help="Adjusts which basic colors are detected (lower range)")
            lower_s = st.slider("Color Intensity (Lower)", 0, 255, int(st.session_state.app.lower_hsv[1]),
                           help="Adjusts how vibrant colors need to be to get detected (lower range)")
            lower_v = st.slider("Brightness (Lower)", 0, 255, int(st.session_state.app.lower_hsv[2]),
                           help="Adjusts how bright colors need to be to get detected (lower range)")
        
            st.subheader("Color Range - Upper Bounds")
            upper_h = st.slider("Color Type (Upper)", 0, 180, int(st.session_state.app.upper_hsv[0]),
                           help="Adjusts which basic colors are detected (upper range)")
            upper_s = st.slider("Color Intensity (Upper)", 0, 255, int(st.session_state.app.upper_hsv[1]),
                           help="Adjusts how vibrant colors can be to get detected (upper range)")
            upper_v = st.slider("Brightness (Upper)", 0, 255, int(st.session_state.app.upper_hsv[2]),
                           help="Adjusts how bright colors can be to get detected (upper range)")
            st.session_state.app.lower_hsv = np.array([lower_h, lower_s, lower_v])
            st.session_state.app.upper_hsv = np.array([upper_h, upper_s, upper_v])
            st.header("AI Analysis")
            load_dotenv()
            if 'GEMINI_API_KEY' not in st.session_state:
                st.session_state.GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')   
            show_api_key = st.checkbox("Configure API Key")
            if show_api_key:
                api_key = st.text_input("Enter Gemini API Key", type="password", 
                                       value=st.session_state.GEMINI_API_KEY)
                st.session_state.GEMINI_API_KEY = api_key
                if st.button("Save API Key"):
                    st.success("API Key saved for this session")
            if st.button("Analyze Drawing"):
                with st.spinner("Analyzing your drawing..."):
                    st.session_state.app.ai_ask_mode = True
                    result = st.session_state.app.generate_analysis(st.session_state.GEMINI_API_KEY)
                    st.session_state.analysis_result = result
                    st.session_state.app.ai_ask_mode = False
            st.header("Advanced Drawing")
            if st.button("Launch Advanced Draw"):
                canvas_save_path = "temp_canvas.png"
                if st.session_state.webcam_started:
                    st.session_state.webcam_started = False
                launched = launch_advanced_draw(canvas_save_path)
                if launched:
                    st.success("Advanced drawing application launched! Check your taskbar or system tray.")
    
    # Custom CSS to ensure larger content fits
    st.markdown("""
    <style>
    .main .block-container {
        max-width: 100%;
        padding: 1rem;
    }
    .stImage > img {
        max-width: 100%;
        height: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.fullscreen_mode:
        if st.button("Exit Full Screen", key="exit_fullscreen", use_container_width=True):
            st.session_state.fullscreen_mode = False
            st.rerun()
        full_col1, full_col2 = st.columns([1, 1])  # Equal width
        with full_col1:
            st.subheader("Camera Feed")
            st.session_state.webcam_placeholder = st.empty()
        with full_col2:
            st.subheader("Drawing Canvas")
            st.session_state.canvas_placeholder = st.empty()
        controls_cols = st.columns(4)
        with controls_cols[0]:
            webcam_control = st.button("Start/Stop Webcam", key="webcam_control_fs", use_container_width=True)
            if webcam_control:
                st.session_state.webcam_started = not st.session_state.webcam_started
                if not st.session_state.webcam_started:
                    st.rerun()
        with controls_cols[1]:
            if st.button("Clear Canvas", key="clear_canvas_fs", use_container_width=True):
                st.session_state.app.clearCanvas()
        with controls_cols[2]:
            color_options = ["BLUE", "GREEN", "RED", "YELLOW"]
            selected_color = st.selectbox("Color", color_options, 
                                         index=st.session_state.app.colorIndex,
                                         key="color_select_fs")
            st.session_state.app.colorIndex = color_options.index(selected_color)
        with controls_cols[3]:
            if st.button("Save Drawing", key="save_drawing_fs", use_container_width=True):
                path = st.session_state.app.save_screenshot()
                with open(path, "rb") as file:
                    st.download_button(
                        label="Download Image",
                        data=file,
                        file_name="virtual_paint_drawing.png",
                        mime="image/png",
                        key="download_btn_fs"
                    )
    else:
        col1, col2 = st.columns([1, 1])  # Equal width
        with col1:
            st.header("Camera Feed")
            st.session_state.webcam_placeholder = st.empty()
            if st.button("Start/Stop Webcam", key="webcam_control_normal"):
                st.session_state.webcam_started = not st.session_state.webcam_started
                if not st.session_state.webcam_started:
                    st.rerun()
        with col2:
            st.header("Drawing Canvas")
            st.session_state.canvas_placeholder = st.empty()
            if st.button("Save Drawing", key="save_drawing_normal"):
                path = st.session_state.app.save_screenshot()
                with open(path, "rb") as file:
                    st.download_button(
                        label="Download Image",
                        data=file,
                        file_name="virtual_paint_drawing.png",
                        mime="image/png",
                        key="download_btn_normal"
                    )
        if st.button("Full Screen Mode (Both Camera & Canvas)", key="full_screen_btn"):
            st.session_state.fullscreen_mode = True
            st.rerun()
    
    if st.session_state.analysis_result and not st.session_state.fullscreen_mode:
        st.header("AI Analysis Result")
        st.markdown(st.session_state.analysis_result)
    
    placeholder_img = np.ones((1200, 1200, 3), dtype=np.uint8) * 200  # Increased to 1200x1200
    cv2.putText(placeholder_img, "Click 'Start Webcam' to begin", (300, 600), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)  # Adjusted position and size
    placeholder_img_rgb = cv2.cvtColor(placeholder_img, cv2.COLOR_BGR2RGB)
    
    paint_window_rgb = cv2.cvtColor(st.session_state.app.paintWindow, cv2.COLOR_BGR2RGB)
    
    if not st.session_state.webcam_started:
        if st.session_state.webcam_placeholder is not None:
            st.session_state.webcam_placeholder.image(placeholder_img_rgb, channels="RGB", use_container_width=True)
        if st.session_state.canvas_placeholder is not None:
            st.session_state.canvas_placeholder.image(paint_window_rgb, channels="RGB", use_container_width=True)
    elif st.session_state.webcam_started:
        # Use WebRTC for webcam access
        ctx = webrtc_streamer(
            key="webcam",
            video_processor_factory=lambda: WebcamProcessor(st.session_state.app),
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if ctx.state.playing:
            # Update canvas display
            paint_window_rgb = cv2.cvtColor(st.session_state.app.paintWindow, cv2.COLOR_BGR2RGB)
            if st.session_state.canvas_placeholder is not None:
                st.session_state.canvas_placeholder.image(paint_window_rgb, channels="RGB", use_container_width=True)
        else:
            if st.session_state.webcam_placeholder is not None:
                st.session_state.webcam_placeholder.image(placeholder_img_rgb, channels="RGB", use_container_width=True)
            st.session_state.webcam_started = False

if __name__ == "__main__":
    run_streamlit_app()
