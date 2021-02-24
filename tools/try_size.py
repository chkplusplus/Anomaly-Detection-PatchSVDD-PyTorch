from PIL import Image 
import numpy as np
img = Image.open('/media/yanglu/data/chkplusplus/projects/data/bottle/ground_truth/broken_large/000_mask.png')
print(np.array(img).shape)