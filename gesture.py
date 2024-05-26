from flask import Flask, render_template_string, Response
from io import BufferedIOBase
from threading import Condition
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder, Quality
from picamera2.outputs import FileOutput
from libcamera import controls, Transform
import cv2
import math
import numpy as np

# HTML 模板
template = '''
    <!DOCTYPE html>
    <html lang="en">
        <body>
            <img src="{{ url_for('video_stream') }}" width="100%">
        </body>
    </html>
    '''

app = Flask(__name__)

# 創建輸出類用於存儲幀
class StreamingOutput(BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

output = StreamingOutput()

# 加載 Haar Cascade 模型
cascade_path = "haarcascade_frontalface_default.xml"  # 使用實際的路徑
face_cascade = cv2.CascadeClassifier(cascade_path)

def adjust_color_balance(img, increase_saturation=True):
    if increase_saturation:
        # 轉換到 HSV 色彩空間增強飽和度
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 50)  # 增加飽和度
        hsv = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def gen_frames():
    while True:
        with output.condition:
            output.condition.wait()
            frame = output.frame

        # 將 JPEG 轉換為 OpenCV 圖像
        img = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        # 調整色彩平衡以改善紫色調
        img = adjust_color_balance(img, increase_saturation=True)

        # 定義感興趣區域並調整到視窗中間
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
        roi = img[center_y-150:center_y+150, center_x-150:center_x+150]
        cv2.rectangle(img, (center_x-150, center_y-150), (center_x+150, center_y+150), (0, 255, 0), 0)
        
        # 更精確的皮膚顏色範圍調整
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 50, 80], dtype=np.uint8)  # 更新飽和度和亮度閾值
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=4)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)
        
        # 尋找輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(cnt, 0.0005 * cv2.arcLength(cnt, True), True)
            hull = cv2.convexHull(cnt)
            
            # 計算凸包和手部區域的面積比
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)
            arearatio = ((areahull - areacnt) / areacnt) * 100

            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)
            l = 0
            
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

                    if angle <= 90:
                        l += 1
                        cv2.circle(roi, far, 3, [255, 0, 0], -1)
                    
                    cv2.line(roi, start, end, [0, 255, 0], 2)
                
            # 根據手指數目顯示不同的信息
            if l == 1:
                if arearatio < 12:
                    text = '0'
                elif arearatio < 17.5:
                    text = 'Best of luck'
                else:
                    text = '1'
            elif l == 2:
                text = '2'
            # 根據檢測到的手勢顯示相應的數字或信息
            cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # 重新編碼為 JPEG 並生成數據流
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
    config = cam.create_video_configuration(
        {'size': (1920, 1080), 'format': 'XBGR8888'},
        transform=Transform(vflip=1),
        controls={'NoiseReductionMode': controls.draft.NoiseReductionModeEnum.HighQuality, 'Sharpness': 1.5}
    )
    cam.configure(config)
    cam.start_recording(JpegEncoder(), FileOutput(output), Quality.VERY_HIGH)

    app.run(host='0.0.0.0')

    cam.stop()