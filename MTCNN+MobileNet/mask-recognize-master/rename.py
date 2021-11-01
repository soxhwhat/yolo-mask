import os
import cv2
import time
after_generate = os.listdir("./data/image/mask")
i = 914
for image in after_generate:
    if image.endswith(".jpg"):
        # # write(image + ";" + str(classes.index(image.split("_")[0])) + "\n")
        # # print(type(image))
        #
        draw = cv2.imread("./data/image/mask/" + image)
        cv2.imwrite("./data/image/mask1/" + "correctmask_" + str(i) + ".jpg", draw)
        i += 1
        time.sleep(0.01)
        print(i)
        # cv2.imshow("999",tt)
        # cv2.waitKey(100)
