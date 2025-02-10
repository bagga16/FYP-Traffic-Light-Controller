
import cv2
import torch
from tkinter import *
from PIL import Image, ImageTk
import threading
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Force the model to use CPU
device = 'cpu'

# Load YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', device=device)
print("Model loaded successfully!")

# Global variables for vehicle counts and traffic control
road1_count = 0
road2_count = 0
current_side = 1
terminate = False
yellow_blink = False
green_blink_state = True

# Detection regions for the videos as polygons
detection_region_road1 = np.array([[230, 180], [380, 185], [590, 300], [20, 300]], np.int32)


detection_region_road2 = np.array([[160, 172], [400, 165], [460, 300], [20, 300]], np.int32)


def is_in_polygon(x, y, polygon):
    """Check if a point (x, y) is inside the specified polygon region."""
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0


def process_video(video_source, canvas, label_text, detection_region, side):
    """Process video frames, count vehicles, and update the UI."""
    global road1_count, road2_count, terminate

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {video_source}")
        return

    while not terminate:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
            continue

        frame = cv2.resize(frame, (600, 400))

        # Perform object detection using YOLOv5
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

        vehicle_count = 0

        for detection in detections:
            x1, y1, x2, y2, conf, cls = map(int, detection[:6])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if is_in_polygon(cx, cy, detection_region):
                vehicle_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.polylines(frame, [detection_region], isClosed=True, color=(0, 0, 255), thickness=2)

        if side == 1:
            road1_count = vehicle_count
        else:
            road2_count = vehicle_count

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.imgtk = imgtk
        canvas.create_image(0, 0, anchor=NW, image=imgtk)

        label_text.set(f"Vehicles Detected on Road {side}:  {vehicle_count}")
        canvas.update()

    cap.release()


def traffic_control():
    """Control traffic signals based on vehicle counts."""
    global current_side, terminate, yellow_blink, green_blink_state

    while not terminate:
        if road1_count > road2_count and current_side != 1:
            yellow_blink = True
            time.sleep(0.5)
            yellow_blink = False
            current_side = 1

        elif road2_count > road1_count and current_side != 2:
            yellow_blink = True
            time.sleep(0.5)
            yellow_blink = False
            current_side = 2

        green_blink_state = not green_blink_state  # Toggle for green "blush"
        time.sleep(0.5)

def create_ui():
    """Create the traffic control UI."""
    global current_side, terminate, yellow_blink, green_blink_state

    root = Tk()
    root.title("Intelligent Traffic Signal Control System")
    root.geometry("1300x900")
    root.configure(bg="#2c3e50")

    # Header
    Label(root, text="Intelligent Traffic Signal Control System", font=("Arial", 26, "bold"), fg="white", bg="#2c3e50").pack(pady=10)

    # Traffic Lights Row
    light_frame = Frame(root, bg="#2c3e50")
    light_frame.place(x=320, y=65)  # Lights positioned at the center top

    # Canvas for Circular Lights
    canvas_road1 = Canvas(light_frame, width=150, height=150, bg="#2c3e50", highlightthickness=0)
    canvas_road1.pack(side=LEFT, padx=0)
    canvas_yellow = Canvas(light_frame, width=150, height=150, bg="#2c3e50", highlightthickness=0)
    canvas_yellow.pack(side=LEFT, padx=210)
    canvas_road2 = Canvas(light_frame, width=150, height=150, bg="#2c3e50", highlightthickness=0)
    canvas_road2.pack(side=LEFT, padx=0)

    def draw_lights():
        """Update circular lights based on the current traffic state."""
        canvas_road1.delete("all")
        canvas_yellow.delete("all")
        canvas_road2.delete("all")

        if yellow_blink:
            canvas_yellow.create_oval(25, 25, 125, 125, fill="yellow", outline="")
        else:
            canvas_yellow.create_oval(25, 25, 125, 125, fill="gray", outline="")

        green_color = "#00FF00" if green_blink_state else "#66FF66"

        if current_side == 1:
            canvas_road1.create_oval(25, 25, 125, 125, fill=green_color, outline="")
            canvas_road2.create_oval(25, 25, 125, 125, fill="red", outline="")
        else:
            canvas_road1.create_oval(25, 25, 125, 125, fill="red", outline="")
            canvas_road2.create_oval(25, 25, 125, 125, fill=green_color, outline="")

        root.after(500, draw_lights)

    draw_lights()

    # Video Frames
    canvas1 = Canvas(root, width=600, height=400, bg="black")
    canvas1.place(x=108, y=230)  # Adjusted position: Top-left
    road1_label_text = StringVar()
    Label(root, textvariable=road1_label_text, font=("Arial", 16), fg="white", bg="#2c3e50").place(x=115, y=670)  # Below video 1

    canvas2 = Canvas(root, width=600, height=400, bg="black")
    canvas2.place(x=804, y=230)  # Adjusted position: Top-right
    road2_label_text = StringVar()
    Label(root, textvariable=road2_label_text, font=("Arial", 16), fg="white", bg="#2c3e50").place(x=812, y=673)  # Below video 2

    # Termination Button at Top-Right
    Button(root, text="Terminate", font=("Arial", 14), fg="white", bg="red", command=lambda: terminate_program(root)).place(x=1400, y=20)

    def terminate_program(root):
        global terminate
        terminate = True
        root.quit()

    threading.Thread(target=traffic_control).start()
    threading.Thread(target=process_video, args=('sample7.mp4', canvas1, road1_label_text, detection_region_road1, 1)).start()
    threading.Thread(target=process_video, args=('sample9.mp4', canvas2, road2_label_text, detection_region_road2, 2)).start()

    root.mainloop()


print("Starting the Intelligent Traffic Signal Control System...")
create_ui()
###








