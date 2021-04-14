import os
import json
import numpy as np
import skimage.draw
import cv2

IMAGE_FOLDER = "/media/yanglu/data/chkplusplus/projects/data/broken_png_1024" # 没用了
MASK_FOLOER = "/media/yanglu/data/chkplusplus/Anomaly-Detection-PatchSVDD-PyTorch/chaguan_dataset/ground_truth/broken_large_4_14/"
PATH_ANNOTATION_JSON = '/media/yanglu/data/chkplusplus/Anomaly-Detection-PatchSVDD-PyTorch/chaguan_dataset/json_file_4_14'
files = [f for f in os.listdir(PATH_ANNOTATION_JSON)]

for file in files:
    # 加载VIA导出的json文件
    file = PATH_ANNOTATION_JSON + "/" + file
    annotations = json.load(open(file, 'r'))
    # print(annotations)
    imgs = annotations["_via_img_metadata"]
    # print(imgs)
    for imgId in imgs:
        imgId = imgId

        print(imgId)
    filename = imgs[imgId]['filename']
    # filename = '33.png'
    regions = imgs[imgId]['regions']
    # if len(regions) <= 0:
    #     continue



    # 图片路径
    # image_path = os.path.join(IMAGE_FOLDER, filename)
    # 读出图片，目的是获取到宽高信息, 不需要了，下边直接用460了
    # print(image_path)
    # image = cv2.imread(image_path)  # image = skimage.io.imread(image_path)
    # height, width = image.shape[:2]
    height, width = 460, 460

    # 创建空的mask
    maskImage = np.zeros((height,width), dtype=np.uint8)

    # 取出第一个标注的类别，本例只标注了一个物件
    s = 0
    while s < len(regions):
        polygons = regions[s]['shape_attributes']
        countOfPoint = len(polygons['all_points_x'])
        points = [None] * countOfPoint
        for i in range(countOfPoint):
            x = int(polygons['all_points_x'][i])
            y = int(polygons['all_points_y'][i])
            points[i] = (x, y)

        contours = np.array(points)
        print(contours)
        # 遍历图片所有坐标
        for i in range(width):
            for j in range(height):
                if cv2.pointPolygonTest(contours, (i, j), False) > 0:
                    maskImage[j,i] = 255
        s = s + 1
    savePath = MASK_FOLOER + filename
    # 保存mask
    cv2.imwrite(savePath, maskImage)
