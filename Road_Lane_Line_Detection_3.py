import dataclasses
from pathlib import Path
import cv2
import numpy as np
# from common import read_cap
from typing import Iterator

'''

### 這隻程式的流程 ###
1. 辨識出車輛以及是否和方格(第34行)重疊
2. 辨識出車道

上述方格 在第150有顯示的方格區域，顏色是淡粉紅色

'''

def main():
    cap = cv2.VideoCapture('imgs\\road_video_Trim.mp4')

    output_folder = Path("imgs\\picture")


    # # 自動建立目錄
    # if output_folder.is_file():
    #     raise NotADirectoryError(output_folder)
    # output_folder.mkdir(parents=True, exist_ok=True)

    # # parents：如果父目錄不存在，是否創建父目錄。
    # # exist_ok：只有在目錄不存在時創建目錄，目錄已存在時不會拋出異常。

    output_counter = 0
    detector = CarDetector(cap, {2, 5, 7})  # {2, 5, 7}為yolo的names，2是小客車，5跟7我忘了，應該是小貨車、大貨車
    area = Area(600, 350, 850, 650)  # 這是辨識方格與汽車是否重疊，我用來顯示的區域
    writer = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        int(cap.get(cv2.CAP_PROP_FPS)),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )


    for frame in read_cap(cap):
        result_img = frame.copy()   # 把原影像分成frame跟result_img，原影像會保持原影像，result_img會拿來做YOLO辨識車輛
        result_img = process_frame(frame, area, detector, result_img)  # YOLO 辨識車輛以及是否和方格重疊
        result_img = detect_move(frame, result_img)  # 辨識車道

        writer.write(result_img)   # 寫成mp4檔案
        result_img = output_result_img(result_img)  # 顯示畫面

        if cv2.waitKey(1) == ord('q'):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


def read_cap(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:  # 讀取的影像的每一個frame是ndarray的data type
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            yield frame

def output_result_img(result_img):
    cv2.imshow('result_img', result_img)

def detect_move(frame, result_img):
    avg = cv2.blur(frame, (4, 4))
    
    # 設定辨識的範圍，這邊試設一個梯形，因為車道通常只會在這個範圍內，所以可以縮減計算量
    lines = process(frame)
    frame = draw_the_lines(frame, lines)  # draw
    result_img = draw_the_lines(result_img, lines)  # draw

    blur = cv2.blur(frame, (4, 4))    # 模糊處理
    diff = cv2.absdiff(avg, blur)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # 篩選出在25到255門檻值的區域
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)


    # morphologyEx : 使用型態轉換函數去除雜訊
    # https://blog.csdn.net/qq_39507748/article/details/104539673
    # https://zhuanlan.zhihu.com/p/496758054
    # 可參考這兩篇
    kernel = np.ones((5, 5), np.uint8)
    # thresh會得到輪廓的資訊
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 產生等高線，輪廓檢測
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_motion = False
    for c in cnts:
        # 忽略太小的區域
        if cv2.contourArea(c) < 5000:
            continue

        has_motion = True
        # 計算等高線的外框範圍    # x，y是矩陣左上點的坐標，w，h是矩陣的寬和高
        (x, y, w, h) = cv2.boundingRect(c)

        # 畫出外框
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 255), 2)  # 圈出車道的範圍而已
        
    # 畫出等高線（除錯用）(黃線 -> 綠線的周圍)
    cv2.drawContours(result_img, cnts, -1, (0, 255, 255), 2)  # 車道的外框線

    return result_img


@dataclasses.dataclass()
class Area:
    x1: int
    y1: int
    x2: int
    y2: int

    def is_overlapped(self, other: "Area") -> bool:    #  : 和 -> 都是 annotation
        if self.x1 > other.x2 or other.x1 > self.x2:
            return False  # 沒有重疊到的
        if self.y1 > other.y2 or other.y1 > self.y2:
            return False  # 沒有重疊到的
        return True  # 剩下有重疊到的


def process_frame(frame: np.ndarray, area: Area, detector: "CarDetector", result_img: np.ndarray) -> np.ndarray:
    # 回傳出這個型態 return value的type n維陣列
    # detect_cars by frame

    # 偵測車輛的範圍，有幾台車就會有幾個area。ex : [Area(x1=836, y1=267, x2=946, y2=351), ...]
    detected_areas = detector.detect(result_img)
    # check detection in or not in the predefined area

    if any(detected_area.is_overlapped(area) for detected_area in detected_areas):  # detected_area跟前面的一樣
        # 如果iterable都為空、0、false，則返回false，如果不都為空、0、false，則返回true。
        cv2.putText(result_img, "Keep a {}".format('Safe Distance'), (580, 500), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 4)
        cv2.line(result_img, (850, 350), (1300, 650), (0, 0, 255), 4)
        cv2.line(result_img, (150, 700), (600, 350), (0, 0, 255), 4)
        cv2.line(result_img, (600, 350), (850, 350), (0, 0, 255), 6)
    else:
        cv2.putText(result_img, "Safe {}".format('Distance'), (625, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.line(result_img, (850, 350), (1300, 650), (0, 255, 255), 4)
        cv2.line(result_img, (150, 700), (600, 350), (0, 255, 255), 4)
        cv2.line(result_img, (600, 350), (850, 350), (0, 255, 255), 6)

    # 對應到34行的area
    cv2.rectangle(result_img, (area.x1, area.y1), (area.x2, area.y2), (128, 128, 255), 6)  # 這是辨識方格與汽車是否重疊，我用來顯示的區域
    return result_img


# Map<Integer> arr = new ArrayList<>()
class CarDetector:
    def __init__(self, cap: cv2.VideoCapture, car_ids: set[int]):  # set為集合，裡面裝int
        self.cap = cap
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.net = cv2.dnn_DetectionModel('yolov4.cfg', 'yolov4.weights')
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.net.setInputParams(size=(608, 608), scale=1 / 255, swapRB=True)
        # self.net.setInputSize((608, 608))
        self.car_ids = car_ids

    def detect(self, frame: np.ndarray) -> list[Area]:
        class_ids, _, boxes = self.net.detect(frame, 0.7, 0.1)  # 偵測汽車 0.7為敏感程度 0.1為幀數
        # boxes為偵測之汽車的陣列
        class_ids = np.array(class_ids)
        print(class_ids, "class_ids")
        print(boxes, "boxes")
        results = []
        for class_id, (x, y, w, h) in zip(class_ids.flatten(), boxes):  # fatten 轉成1維陣列
            # 可以是n個 zip會衍生一個疊帶系統，會把第一個全拿出來，tuple，第二次也是包成tuple，當其中任何一個疊帶器停止，整個zip就會停掉
            # 假設有A，B兩個list A長度為5，B長度為7，只會跑5次疊帶。
            # 把n個iterable的第M個元素包成一個tuple，M為長度最小的iterable長度
            if class_id in self.car_ids:
                results.append(area := Area(x, y, x + w, y + h))
                cv2.rectangle(frame, (area.x1, area.y1), (area.x2, area.y2), (255, 128, 64), 3)
        return results


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_the_lines(img, lines):  # 畫綠線
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)  # 粗細

    img = cv2.addWeighted(img, 1, blank_image, 1, 0.0)  # 第二個為相片黑白色調整 1為正常
    return img


def process(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [  # 設區域
        (0, height-50),
        (width / 3, height / 1.85),  #  / 1.8
        (width / 14 * 11, height / 1.85),  #  / 1.8
        (width, height - 20),
        (width-200, height-200)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image,
                                       np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi / 180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=250)
    return lines

if __name__ == '__main__':
    main()
