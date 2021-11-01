import os
import time
import cv2
import numpy as np
from keras.applications.imagenet_utils import preprocess_input

import utils as utils
from net.mobilenet import MobileNet
from mtcnn import mtcnn


class face_rec():
    def __init__(self):
        self.mtcnn_model = mtcnn()  # 创建mtcnn的模型  用于检测人脸
        self.threshold = [0.5,0.6,0.8]  # 三个神经网络的门限 Pnet Rnet Onet

        self.classes_path = "model_data/classes.txt"
        self.class_names = self._get_class()
        self.Crop_HEIGHT = 160  # 设置输入图像的宽高
        self.Crop_WIDTH = 160
        self.NUM_CLASSES = len(self.class_names) # 获取分类个数
        print("num=",self.NUM_CLASSES)

        #  创建mobilenet的模型 用于判断是否佩戴口罩
        self.mask_model = MobileNet(input_shape=[self.Crop_HEIGHT,self.Crop_WIDTH,3], classes=self.NUM_CLASSES)  # Mobilenet 输入图像为正方形
        self.mask_model.load_weights("./logs/ep071-loss0.013-val_loss0.041.h5")

    def _get_class(self):  # 获取分类的名字
        classes_path = os.path.expanduser(self.classes_path)  # 获取绝对路径
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def recognize(self,draw):  # 检测人脸 识别人脸类型
        height,width,_ = np.shape(draw) # 获取摄像头的图像宽高
        draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)  # 检测人脸
        if len(rectangles)==0:
            return
        # print(len(rectangles))
        # print(rectangles.ndim)
        # print(rectangles)
        rectangles = np.array(rectangles,dtype=np.int32)
        rectangles_temp = utils.rect2square(rectangles)  # 将长方形转正方形
        rectangles_temp[:, [0,2]] = np.clip(rectangles_temp[:, [0,2]], 0, width)  # 让框不越界
        rectangles_temp[:, [1,3]] = np.clip(rectangles_temp[:, [1,3]], 0, height)

        #  对检测到的人脸进行编码
        classes_all = []
        for rectangle in rectangles_temp:
            #   截取图像
            landmark = np.reshape(rectangle[5:15], (5,2)) - np.array([int(rectangle[0]), int(rectangle[1])])
            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]

            #   利用人脸关键点进行人脸对齐
            crop_img,_ = utils.Alignment_1(crop_img,landmark)

            crop_img = cv2.resize(crop_img, (self.Crop_WIDTH,self.Crop_HEIGHT))
            crop_img = preprocess_input(np.reshape(np.array(crop_img, np.float64),[1, self.Crop_HEIGHT, self.Crop_WIDTH, 3]))

            classes = self.class_names[np.argmax(self.mask_model.predict(crop_img)[0])]
            classes_all.append(classes)

        rectangles = rectangles[:, 0:4]
        #  画框
        for (left, top, right, bottom), c in zip(rectangles,classes_all):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(draw, c, (left , bottom - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 255, 0), 2)
        return draw

if __name__ == "__main__":
    pTime = 0
    cTime = 0
    img = face_rec()  # 从类中得到结果图像
    print("88888888888888888")
    camera_number = 0
    video_capture = cv2.VideoCapture(camera_number)  # 打开摄像头读入数据流
    # time.sleep(1)
    while True:
        ret, draw = video_capture.read() # draw：每帧图片 ret：是否读出
        if ret:
            draw = cv2.flip(draw, 1)
            img.recognize(draw)
            cTime = time.time()
            fps = 1 / (cTime - pTime)  # fps的计算
            fps = fps * 100
            fps = int(fps) / 100
            pTime = cTime
            cv2.putText(draw, "FPS " + str(float(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            draw = cv2.resize(draw, (draw.shape[1] * 2, draw.shape[0] * 2))         # 图像放大 img.shape[0]:高 rows
            cv2.imshow('Video', draw)
            C = cv2.waitKey(1)
            if C & 0xFF == 27: # 按 ”ESC“退出
                break
            elif C & 0xFF == ord('q'):
                cv2.imwrite("./test001.jpg",draw)
                print("Good")
            elif C & 0xFF == ord('w'):
                cv2.imwrite("./test002.jpg", draw)
                print("Good")
            elif C & 0xFF == ord('e'):
                cv2.imwrite("./test003.jpg", draw)
                print("Good")
        else:
            print("打开相机失败")
    video_capture.release()  # 摄像头类释放
    cv2.destroyAllWindows()