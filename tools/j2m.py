import os
import numpy as np
import cv2 as cv
import json
import skimage
dataset_dir="/media/yanglu/data/chkplusplus/Anomaly-Detection-PatchSVDD-PyTorch/json"
# subset="spalling"
json_path="2_8_json.json"
output_path = "/media/yanglu/data/chkplusplus/projects/data/1024_png_mask"
# dataset_dir = os.path.join(dataset_dir, subset)
# We mostly care about the x and y coordinates of each region
annotations = json.load(open(os.path.join(dataset_dir, json_path)))#加载json文件
#json文件中字典嵌套字典，字典嵌套列表。为了取到关键的x，y点的数据，得一一层把字典，列表剥开。
annotations = list(annotations.values())
annotations_point=annotations[0]
annotations_point=list(annotations_point.values())
 
annotations_point = [a for a in annotations_point if a['regions']]
 
for i in range(len(annotations_point)):#遍历每一张图的信息
 
    filename=annotations_point[i]['filename']
    mask = np.zeros([1024, 1024, 3],dtype=np.uint8)#设置输出图片维度大小
    for j in range(len(annotations_point[i]['regions'])):#一张图可能存在多个标注区域
#json文件中字典嵌套字典，字典嵌套列表。为了取到关键的x，y点的数据，得一一层把字典，列表剥开。
        point = annotations_point[i]['regions'][j].values()[0]
        point_x=point['all_points_x']#提取x点数据
        point_y=point['all_points_y']#提取y点数据
        rr, cc = skimage.draw.polygon(point_y, point_x)#绘制轮廓，得到轮廓内所有点的坐标
        mask[rr, cc, 0] = 255#填充颜色，255 0 0：蓝色；0 255 0：绿色；0 0 255：红色
    #cv.imshow("mask_"+filename,mask)#调试时显示查看
    #cv.waitKey(0)
    cv.imwrite(output_path+filename,mask)#保存png格式标签