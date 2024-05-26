from flask import Flask, render_template_string, Response
from io import BufferedIOBase
from threading import Condition
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder, Quality
from picamera2.outputs import FileOutput
from libcamera import controls, Transform
import cv2
import numpy as np
import math

# HTML Template for displaying the video stream
template = '''
<!DOCTYPE html>
<html lang="en">
    <body>
        <img src="{{ url_for('video_stream') }}" width="100%">
    </body>
</html>
'''

app = Flask(__name__)

class StreamingOutput(BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

output = StreamingOutput()

def simple_white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    img = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return img

def gen_frames():
    while True:
        with output.condition:
            output.condition.wait()
            frame = output.frame

        img = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = simple_white_balance(img)
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

        roi = img[center_y-150:center_y+150, center_x-150:center_x+150]
        cv2.rectangle(img, (center_x-150, center_y-150), (center_x+150, center_y+150), (0, 255, 0), 2)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=4)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        text = "No gesture detected"
        if len(contours) > 0:
            cnt = max(contours, key=lambda x: cv2.contourArea(x))
            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) > 0:
                hull = cv2.convexHull(approx, returnPoints=True)
                if len(hull) > 0:
                    areahull = cv2.contourArea(hull)
                    areacnt = cv2.contourArea(cnt)
                    arearatio = (areahull - areacnt) / areacnt * 100
                    defects = cv2.convexityDefects(approx, cv2.convexHull(approx, returnPoints=False))

                    if defects is not None:
                        num_defects = 0
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(approx[s][0])
                            end = tuple(approx[e][0])
                            far = tuple(approx[f][0])
                            angle = math.acos(min(1, max(-1, (np.linalg.norm(np.subtract(start, far))**2 + np.linalg.norm(np.subtract(end, far))**2 - np.linalg.norm(np.subtract(start, end))**2) / (2 * np.linalg.norm(np.subtract(start, far)) * np.linalg.norm(np.subtract(end, far)))))) * 57

                            if angle <= 90:
                                num_defects += 1

                        if num_defects == 0 or num_defects == 1:
                            text = 'Rock'
                        elif num_defects == 2:
                            text = 'Scissors'
                        elif num_defects >= 3:
                            text = 'Paper'

        cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        _, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route("/", methods=['GET'])
def get_stream_html():
    return render_template_string(template)

@app.route('/api/stream')
def video_stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    cam = Picamera2()
    config = cam.create_video_configuration({'size': (1920, 1080), 'format': 'XBGR8888'},
                                            transform=Transform(vflip=1),
                                            controls={'NoiseReductionMode': controls.draft.NoiseReductionModeEnum.HighQuality, 'Sharpness': 1.5})
    cam.configure(config)
    cam.start_recording(JpegEncoder(), FileOutput(output), Quality.VERY_HIGH)
    app.run(host='0.0.0.0')
    cam.stop()
