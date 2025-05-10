import streamlit as st
import cv2
import numpy as np
import time
import socket
import struct
import json
import pickle
import threading
import queue
from PIL import Image

class StreamlitClient:
    """Streamlit client for the Smart Air Drawing System."""
    def __init__(self, host='localhost', port=5555):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.running = False
        
        # Data storage
        self.current_frame = None
        self.current_canvas = None
        self.current_mode = "draw"
        self.current_letter = "D"
        self.current_message = "Connecting to server..."
        self.recognition_results = {
            "text": [],
            "shapes": [],
            "math": [],
            "voice": []
        }
        
        # Threading
        self.receive_thread = None
        self.data_lock = threading.Lock()
        self.result_queue = queue.Queue(maxsize=10)
        
    def connect(self):
        """Connect to the backend server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            st.success(f"Connected to server at {self.host}:{self.port}")
            
            # Start receive thread
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_thread)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            return True
        except Exception as e:
            st.error(f"Connection error: {e}")
            self.connected = False
            return False
            
    def disconnect(self):
        """Disconnect from the backend server."""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        st.warning("Disconnected from server")
        
    def send_command(self, command):
        """Send a command to the backend server."""
        if not self.connected:
            st.warning("Not connected to server")
            return False
            
        try:
            # Serialize command
            command_json = json.dumps(command).encode('utf-8')
            msg_len = struct.pack('!I', len(command_json))
            
            # Send command
            self.socket.sendall(msg_len + command_json)
            return True
        except Exception as e:
            st.error(f"Send error: {e}")
            self.connected = False
            return False
            
    def _receive_thread(self):
        """Thread that receives data from the server."""
        self.socket.settimeout(1.0)  # 1 second timeout for recv
        
        while self.running:
            try:
                # Receive message length (4 bytes)
                msg_len_bytes = self.socket.recv(4)
                if not msg_len_bytes:
                    break
                    
                msg_len = struct.unpack('!I', msg_len_bytes)[0]
                
                # Receive message data
                data = b''
                while len(data) < msg_len:
                    chunk = self.socket.recv(min(4096, msg_len - len(data)))
                    if not chunk:
                        break
                    data += chunk
                    
                if len(data) < msg_len:
                    break
                    
                # Decode and process result
                result = pickle.loads(data)
                self._process_result(result)
                
                # Put result in queue for UI thread
                try:
                    self.result_queue.put(result, block=False)
                except queue.Full:
                    # If queue is full, remove oldest item and try again
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put(result, block=False)
                    except:
                        pass
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Receive error: {e}")
                break
                
        self.connected = False
        print("Receive thread stopped")
        
    def _process_result(self, result):
        """Process a result from the server."""
        with self.data_lock:
            # Convert JPEG frames back to numpy arrays
            if 'frame' in result and isinstance(result['frame'], bytes):
                frame_array = np.frombuffer(result['frame'], dtype=np.uint8)
                self.current_frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
            if 'canvas' in result and isinstance(result['canvas'], bytes):
                canvas_array = np.frombuffer(result['canvas'], dtype=np.uint8)
                self.current_canvas = cv2.imdecode(canvas_array, cv2.IMREAD_COLOR)
                
            # Update other data
            if 'mode' in result:
                self.current_mode = result['mode']
                
            if 'letter' in result:
                self.current_letter = result['letter']
                
            if 'message' in result:
                self.current_message = result['message']
                
            if 'recognition_results' in result:
                self.recognition_results = result['recognition_results']
                
    def set_mode(self, mode):
        """Set the current drawing mode."""
        command = {
            'type': 'set_mode',
            'data': {'mode': mode}
        }
        return self.send_command(command)
        
    def set_letter(self, letter):
        """Set the current letter."""
        command = {
            'type': 'set_letter',
            'data': {'letter': letter}
        }
        return self.send_command(command)
        
    def set_color(self, color):
        """Set the current drawing color."""
        command = {
            'type': 'set_color',
            'data': {'color': color}
        }
        return self.send_command(command)
        
    def toggle_template(self, show_template):
        """Toggle the template visibility."""
        command = {
            'type': 'toggle_template',
            'data': {'show_template': show_template}
        }
        return self.send_command(command)
        
    def clear_canvas(self):
        """Clear the canvas."""
        command = {
            'type': 'clear_canvas',
            'data': {}
        }
        return self.send_command(command)
        
    def save_drawing(self, filename=None):
        """Save the current drawing to a file."""
        command = {
            'type': 'save_drawing',
            'data': {'filename': filename}
        }
        return self.send_command(command)
        
    def start_voice(self):
        """Start voice recognition."""
        command = {
            'type': 'start_voice',
            'data': {}
        }
        return self.send_command(command)
        
    def toggle_challenge(self):
        """Toggle challenge mode."""
        command = {
            'type': 'toggle_challenge',
            'data': {}
        }
        return self.send_command(command)
        
    def get_latest_result(self):
        """Get the latest result from the queue."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None


def main():
    st.set_page_config(
        page_title="Smart Air Drawing System",
        page_icon="✏️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Smart Air Drawing System")
    
    # Initialize session state
    if 'client' not in st.session_state:
        st.session_state.client = StreamlitClient()
        
    if 'connected' not in st.session_state:
        st.session_state.connected = False
        
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
        
    # Sidebar
    st.sidebar.title("Smart Air Drawing")
    
    # Connection settings
    with st.sidebar.expander("Connection Settings", expanded=not st.session_state.connected):
        host = st.text_input("Server Host", value="localhost")
        port = st.number_input("Server Port", value=5555, min_value=1, max_value=65535)
        
        if not st.session_state.connected:
            if st.button("Connect to Server"):
                st.session_state.client = StreamlitClient(host=host, port=port)
                st.session_state.connected = st.session_state.client.connect()
        else:
            if st.button("Disconnect"):
                st.session_state.client.disconnect()
                st.session_state.connected = False
    
    # Mode selection
    if st.session_state.connected:
        st.sidebar.subheader("Drawing Mode")
        mode_options = ["draw", "text", "shape", "math", "voice"]
        selected_mode = st.sidebar.selectbox(
            "Select Mode", 
            mode_options,
            index=mode_options.index(st.session_state.client.current_mode)
        )
        if selected_mode != st.session_state.client.current_mode:
            st.session_state.client.set_mode(selected_mode)
        
        # Letter selection for draw mode
        if st.session_state.client.current_mode == "draw":
            st.sidebar.subheader("Letter Selection")
            letter_options = ['D', 'R', 'O', 'W']
            selected_letter = st.sidebar.selectbox(
                "Select Letter", 
                letter_options,
                index=letter_options.index(st.session_state.client.current_letter)
            )
            if selected_letter != st.session_state.client.current_letter:
                st.session_state.client.set_letter(selected_letter)
        
        # Template toggle
        st.sidebar.subheader("Display Options")
        show_template = st.sidebar.checkbox("Show Template", value=True)
        if show_template != st.session_state.client.show_template:
            st.session_state.client.toggle_template(show_template)
        
        # Action buttons
        st.sidebar.subheader("Actions")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("Clear Canvas"):
                st.session_state.client.clear_canvas()
        
        with col2:
            if st.button("Save Drawing"):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"air_drawing_{timestamp}.png"
                st.session_state.client.save_drawing(filename)
                st.sidebar.success(f"Drawing saved as {filename}")
        
        col3, col4 = st.sidebar.columns(2)
        
        with col3:
            if st.button("Voice Recognition"):
                st.session_state.client.start_voice()
        
        with col4:
            if st.button("Challenge Mode"):
                st.session_state.client.toggle_challenge()
    
    # Main content
    if st.session_state.connected:
        # Status message
        st.markdown(f"**Status:** {st.session_state.client.current_message}")
        
        # Mode indicator
        st.markdown(f"**Current Mode:** {st.session_state.client.current_mode.upper()}")
        
        # Camera toggle
        camera_col, placeholder_col = st.columns([1, 3])
        with camera_col:
            camera_active = st.toggle("Activate Camera", value=st.session_state.camera_active)
            if camera_active != st.session_state.camera_active:
                st.session_state.camera_active = camera_active
        
        # Main display area
        display_col1, display_col2 = st.columns(2)
        
        # Get latest result
        result = st.session_state.client.get_latest_result()
        if result:
            # Update client data
            st.session_state.client._process_result(result)
        
        with display_col1:
            st.subheader("Camera Feed")
            camera_placeholder = st.empty()
            if st.session_state.camera_active:
                with st.session_state.client.data_lock:
                    if st.session_state.client.current_frame is not None:
                        # Convert to RGB for Streamlit
                        frame_rgb = cv2.cvtColor(st.session_state.client.current_frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    else:
                        camera_placeholder.info("Waiting for camera feed...")
            else:
                camera_placeholder.info("Camera is inactive. Toggle 'Activate Camera' to start.")
        
        with display_col2:
            st.subheader("Drawing Canvas")
            canvas_placeholder = st.empty()
            with st.session_state.client.data_lock:
                if st.session_state.client.current_canvas is not None:
                    # Convert to RGB for Streamlit
                    canvas_rgb = cv2.cvtColor(st.session_state.client.current_canvas, cv2.COLOR_BGR2RGB)
                    canvas_placeholder.image(canvas_rgb, channels="RGB", use_column_width=True)
                else:
                    canvas_placeholder.info("Waiting for canvas...")
        
        # Recognition results
        with st.session_state.client.data_lock:
            if st.session_state.client.current_mode == "text" and st.session_state.client.recognition_results["text"]:
                st.subheader("Recognized Text")
                text_result = "".join([t[0] for t in st.session_state.client.recognition_results["text"]])
                st.markdown(f"**Text:** {text_result}")
            
            elif st.session_state.client.current_mode == "shape" and st.session_state.client.recognition_results["shapes"]:
                st.subheader("Recognized Shapes")
                for i, (shape, _, _) in enumerate(st.session_state.client.recognition_results["shapes"][-5:]):
                    st.markdown(f"{i+1}. {shape.capitalize()}")
            
            elif st.session_state.client.current_mode == "math" and st.session_state.client.recognition_results["math"]:
                st.subheader("Math Expressions")
                for expr, _ in st.session_state.client.recognition_results["math"][-3:]:
                    st.latex(expr)
            
            elif st.session_state.client.current_mode == "voice" and st.session_state.client.recognition_results["voice"]:
                st.subheader("Voice Notes")
                for note, _ in st.session_state.client.recognition_results["voice"][-5:]:
                    st.markdown(f"- {note}")
    else:
        st.info("Please connect to the server using the sidebar.")
    
    # Instructions
    with st.expander("Instructions", expanded=False):
        st.markdown("""
        ### How to use Smart Air Drawing
        
        1. **Connect to the server** using the sidebar
        2. **Activate the camera** using the toggle button
        3. **Select a mode** from the sidebar:
           - **Draw**: Practice drawing letters in the air
           - **Text**: Convert your air drawings to text
           - **Shape**: Draw and recognize shapes
           - **Math**: Create mathematical expressions
           - **Voice**: Add voice notes (requires microphone)
        4. **Use hand gestures**:
           - Index finger extended: Draw
           - Middle finger extended: Switch mode
           - All fingers extended: Clear canvas
        5. **Adjust settings** in the sidebar
        6. **Save your drawing** when finished
        """)


if __name__ == "__main__":
    main()
