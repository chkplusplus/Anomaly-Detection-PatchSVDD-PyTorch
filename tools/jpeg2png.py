from PIL import Image 
import glob, os
directory = "/media/yanglu/data/chkplusplus/projects/data/broken_png"
for infile in glob.glob("/media/yanglu/data/chkplusplus/projects/data/broken_large/*.JPG"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    rgb_im = im.convert('RGB')
    rgb_im.save(directory + file + ".png", "PNG")
for infile in glob.glob("/media/yanglu/data/chkplusplus/projects/data/broken_large/*.jpg"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    rgb_im = im.convert('RGB')
    rgb_im.save(directory + file + ".png", "PNG") 
for infile in glob.glob("/media/yanglu/data/chkplusplus/projects/data/broken_large/*.JPEG"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    rgb_im = im.convert('RGB')
    rgb_im.save(directory + file + ".png", "PNG")
for infile in glob.glob("*.jpeg"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    rgb_im = im.convert('RGB')
    rgb_im.save(directory + file + ".png", "PNG")