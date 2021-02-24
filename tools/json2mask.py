import os
import json
import numpy as np
import skimage.draw
import cv2

IMAGE_FOLDER = "/media/yanglu/data/chkplusplus/projects/data/broken_png_1024"
MASK_FOLOER = "/media/yanglu/data/chkplusplus/projects/data/1024_png_mask/"
PATH_ANNOTATION_JSON = '17_7.json'

# 加载VIA导出的json文件
annotations = json.load(open(PATH_ANNOTATION_JSON, 'r'))
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
image_path = os.path.join(IMAGE_FOLDER, filename)
# 读出图片，目的是获取到宽高信息
# print(image_path)
# image = cv2.imread(image_path)  # image = skimage.io.imread(image_path)
# height, width = image.shape[:2]
height, width = 1024, 1024

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

# polygons1 = regions[1]['shape_attributes']

# 图片路径
# image_path = os.path.join(IMAGE_FOLDER, filename)
# 读出图片，目的是获取到宽高信息
# print(image_path)
# image = cv2.imread(image_path)  # image = skimage.io.imread(image_path)
# height, width = image.shape[:2]
# height, width = 1024, 1024

# 创建空的mask
# maskImage = np.zeros((height,width), dtype=np.uint8)
# countOfPoints1 = len(polygons1['all_points_x'])
# points1 = [None] * countOfPoints1
# for i in range(countOfPoints1):
#     x = int(polygons1['all_points_x'][i])
#     y = int(polygons1['all_points_y'][i])
#     points1[i] = (x, y)

# contours1 = np.array(points)
# print(contours)
# # 遍历图片所有坐标
# for i in range(width):
#     for j in range(height):
#         if cv2.pointPolygonTest(contours1, (i, j), False) > 0:
#             maskImage[j,i] = 255
savePath = MASK_FOLOER + filename
# 保存mask
cv2.imwrite(savePath, maskImage)