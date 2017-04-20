import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
image = mnist[0].images
label = mnist[0].labels

image = np.reshape(image, [len(image), 28, 28])

def generateOneImage(size,minLen,tgtNum):
    posList = []
    while len(posList)<=tgtNum:
        pos = (np.random.randint(0,len(image)),
               np.random.randint(0,size[0]),
               np.random.randint(0,size[1]),
               np.random.uniform(-30.,+30))
        isGood = True
        for a in posList:
            if np.sqrt( (pos[1]-a[1])**2 + (pos[2]-a[2])**2) < minLen: isGood=False
        if not isGood : continue
        posList.append(pos)

    res = Image.fromarray(np.zeros(size,dtype=np.float32))

    for a in posList:
        idx,x,y,r = a
        img  = Image.fromarray(image[idx]*255.)
        img  = img.rotate(r)
        res.paste(img,(int(x-14),int(y-14),int(x+14),int(y+14)))

    #res.show()
    return np.asarray(res)

res = generateOneImage((500,500),50,10)
#print(res.sum())
