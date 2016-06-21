import os, numpy, PIL
from PIL import Image
import cv2

# Access all PNG files in directory
allfiles=os.listdir("/media/Secuencias/kitti/object/training/image_2")
imlist=[filename for filename in allfiles if  filename[-4:] in [".png",".PNG"]]
#print imlist

# Assuming all images are the same size, get dimensions of first image
w,h=Image.open("/media/Secuencias/kitti/object/training/image_2/" + imlist[0]).size
h=376
N=len(imlist)

# Create a numpy array of floats to store the average (assume RGB images)
arr=numpy.zeros((h,w,3),numpy.float)

b_mean = 0
g_mean = 0
r_mean = 0
# Build up average pixel intensities, casting each image as an array of floats
i = 0
print N, "images"
for im in imlist:
    i+=1
    print (i*100)/N
    imarr=numpy.array(Image.open("/media/Secuencias/kitti/object/training/image_2/"+im),dtype=numpy.float)
    #cv2.imshow("skd", imarr)
    #cv2.waitKey(0)
    b_mean += imarr[2].sum()/(imarr[2].size*N)
    g_mean += imarr[1].sum()/(imarr[1].size*N)
    r_mean += imarr[0].sum()/(imarr[0].size*N)
    zeros = numpy.zeros((h,w,3), dtype=numpy.int32)
    #print imarr.shape[0]
    #print imarr.shape[1]
    zeros[:imarr.shape[0], :imarr.shape[1], :] = imarr
    arr=arr+zeros/N

# Round values in array and cast as 8-bit integer
arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

print 'b_mean ', b_mean
print 'g_mean ', g_mean
print 'r_mean ', r_mean

# Generate, save and preview final image
out=Image.fromarray(arr,mode="RGB")
out.save("Average.png")
#out.show()
