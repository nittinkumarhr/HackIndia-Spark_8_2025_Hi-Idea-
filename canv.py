import cv2
import numpy as np

# Canvas setup (larger initial size to avoid resizing)
canvas_width, canvas_height = 1200, 600  # Increased to accommodate longer text
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # White background


default_text = "Eat Thank You Goodbye"  
initial_font_size = 2  # OpenCV font scale (min 1, max 5)
font = cv2.FONT_HERSHEY_SIMPLEX
initial_font_color = (0, 0, 0)  # Black text (BGR format)
thickness = 3
initial_speed = 5  # Frames per brushstroke (min 1, max 20)

# Load text from file or use default
def load_text_from_file(filename="input.txt"):
    try:
        with open(filename, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"File {filename} not found. Using default text.")
        return default_text

text = load_text_from_file()

# Helper function to update text positions and ensure it fits within canvas
def update_positions(text, font, font_size, thickness, canvas_width):
    char_positions = []
    total_width = sum(cv2.getTextSize(char, font, font_size, thickness)[0][0] for char in text)
    x_pos = max(50, (canvas_width - total_width) // 2)  # Center text, ensure it doesn’t go below 50
    y_base = canvas_height // 2
    for char in text:
        (w, h), _ = cv2.getTextSize(char, font, font_size, thickness)
        if x_pos + w <= canvas_width:  # Only add if it fits
            char_positions.append((x_pos, y_base + h // 2))
            x_pos += w
        else:
            break  # Stop if text would exceed canvas width
    return char_positions

# Mouse callback function to handle button clicks
def mouse_callback(event, x, y, flags, param):
    global font_size, brush_speed, font_color, char_positions, brush_progress, char_index, drawing_complete
    if event == cv2.EVENT_LBUTTONDOWN:
        # Increase Size Button (10, 70, 100, 100)
        if 10 <= x <= 100 and 70 <= y <= 100:
            font_size = min(font_size + 0.5, 5)
            char_positions = update_positions(text, font, font_size, thickness, canvas_width)
            if not drawing_complete:
                brush_progress.clear()
                char_index = 0
        # Decrease Size Button (110, 70, 200, 100)
        elif 110 <= x <= 200 and 70 <= y <= 100:
            font_size = max(font_size - 0.5, 1)
            char_positions = update_positions(text, font, font_size, thickness, canvas_width)
            if not drawing_complete:
                brush_progress.clear()
                char_index = 0
        # Increase Speed Button (210, 70, 300, 100)
        elif 210 <= x <= 300 and 70 <= y <= 100:
            brush_speed = max(brush_speed - 1, 1)
        # Decrease Speed Button (310, 70, 400, 100)
        elif 310 <= x <= 400 and 70 <= y <= 100:
            brush_speed = min(brush_speed + 1, 20)
        # Red Button (410, 70, 440, 100)
        elif 410 <= x <= 440 and 70 <= y <= 100:
            font_color = (0, 0, 255)
            if not drawing_complete:
                brush_progress.clear()
                char_index = 0
        # Green Button (450, 70, 480, 100)
        elif 450 <= x <= 480 and 70 <= y <= 100:
            font_color = (0, 255, 0)
            if not drawing_complete:
                brush_progress.clear()
                char_index = 0
        # Blue Button (490, 70, 520, 100)
        elif 490 <= x <= 520 and 70 <= y <= 100:
            font_color = (255, 0, 0)
            if not drawing_complete:
                brush_progress.clear()
                char_index = 0

# Drawing function with painting effect, progress bars, and clickable buttons
def draw_text_with_painting_effect(text):
    global font_size, brush_speed, font_color, char_positions, brush_progress, char_index, drawing_complete
    running = True
    frame_count = 0
    char_index = 0
    brush_progress = {}
    font_size = initial_font_size
    brush_speed = initial_speed
    font_color = initial_font_color
    drawing_complete = False

    # Initial positions
    char_positions = update_positions(text, font, font_size, thickness, canvas_width)

    # Set up mouse callback
    cv2.namedWindow("Text Drawing")
    cv2.setMouseCallback("Text Drawing", mouse_callback)

    while running:
        # Create a fresh canvas
        temp_canvas = canvas.copy()

        # Handle exit via Esc key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc to exit
            running = False

        # Automate painting (only if not complete)
        if not drawing_complete and char_index < len(char_positions):
            if char_index not in brush_progress:
                brush_progress[char_index] = 0
            if frame_count % brush_speed == 0:
                brush_progress[char_index] += 1

        # Draw each character with painting effect
        for i, (x, y) in enumerate(char_positions):
            if i in brush_progress:
                char = text[i]
                (w, h), _ = cv2.getTextSize(char, font, font_size, thickness)
                progress = min(brush_progress[i], w)
                temp_img = np.ones_like(temp_canvas) * 255
                cv2.putText(temp_img, char, (x, y), font, font_size, font_color, thickness)
                temp_canvas[:, x:x + progress] = temp_img[:, x:x + progress]
            if i == char_index and brush_progress.get(i, 0) >= cv2.getTextSize(text[i], font, font_size, thickness)[0][0]:
                char_index += 1

        # Draw cursor (only during drawing)
        if char_index < len(char_positions) and not drawing_complete:
            cursor_x = char_positions[char_index][0]
            cursor_y = char_positions[char_index][1]
            if frame_count % 20 < 10:  # Blinking effect
                cv2.putText(temp_canvas, "|", (cursor_x, cursor_y), font, font_size, (150, 150, 150), thickness)
        elif not drawing_complete:
            drawing_complete = True

        # Draw progress bars
        # Size Progress Bar (1 to 5)
        cv2.putText(temp_canvas, "Size:", (10, 30), font, 0.7, (255, 0, 0), 2)
        bar_width = 100
        bar_height = 10
        size_progress = int((font_size - 1) / 4 * bar_width)
        cv2.rectangle(temp_canvas, (70, 20), (70 + bar_width, 20 + bar_height), (200, 200, 200), -1)
        cv2.rectangle(temp_canvas, (70, 20), (70 + size_progress, 20 + bar_height), (0, 255, 255), -1)

        # Speed Progress Bar (1 to 20, inverted)
        cv2.putText(temp_canvas, "Speed:", (10, 50), font, 0.7, (255, 0, 0), 2)
        speed_progress = int((20 - brush_speed) / 19 * bar_width)
        cv2.rectangle(temp_canvas, (70, 40), (70 + bar_width, 40 + bar_height), (200, 200, 200), -1)
        cv2.rectangle(temp_canvas, (70, 40), (70 + speed_progress, 40 + bar_height), (0, 255, 255), -1)

        # Draw on-screen buttons
        # Increase Size
        cv2.rectangle(temp_canvas, (10, 70), (100, 100), (200, 200, 200), -1)
        cv2.putText(temp_canvas, "Size +", (25, 90), font, 0.5, (0, 0, 0), 1)

        # Decrease Size
        cv2.rectangle(temp_canvas, (110, 70), (200, 100), (200, 200, 200), -1)
        cv2.putText(temp_canvas, "Size -", (125, 90), font, 0.5, (0, 0, 0), 1)

        # Increase Speed
        cv2.rectangle(temp_canvas, (210, 70), (300, 100), (200, 200, 200), -1)
        cv2.putText(temp_canvas, "Speed +", (220, 90), font, 0.5, (0, 0, 0), 1)

        # Decrease Speed
        cv2.rectangle(temp_canvas, (310, 70), (400, 100), (200, 200, 200), -1)
        cv2.putText(temp_canvas, "Speed -", (320, 90), font, 0.5, (0, 0, 0), 1)

        # Color Buttons
        cv2.rectangle(temp_canvas, (410, 70), (440, 100), (0, 0, 255), -1)  # Red
        cv2.putText(temp_canvas, "R", (420, 90), font, 0.5, (255, 255, 255), 1)
        
        cv2.rectangle(temp_canvas, (450, 70), (480, 100), (0, 255, 0), -1)  # Green
        cv2.putText(temp_canvas, "G", (460, 90), font, 0.5, (0, 0, 0), 1)
        
        cv2.rectangle(temp_canvas, (490, 70), (520, 100), (255, 0, 0), -1)  # Blue
        cv2.putText(temp_canvas, "B", (500, 90), font, 0.5, (255, 255, 255), 1)

        # Show the canvas
        cv2.imshow("Text Drawing", temp_canvas)
        frame_count += 1

        # Stop after completion and wait for exit
        if drawing_complete:
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Esc to exit
                    running = False
                    break

    cv2.destroyAllWindows()

# Main function
def main():
    draw_text_with_painting_effect(text)

# Run the program
if __name__ == "__main__":
    main()