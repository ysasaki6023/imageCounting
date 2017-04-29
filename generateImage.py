import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os,argparse
import collections,sys,csv,shutil,random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
image = mnist[0].images
label = mnist[0].labels

image = np.reshape(image, [len(image), 28, 28])

class sampleGen:
    def __init__(self,size,minLen,tgtNum):
        self.size = size
        self.minLen = minLen
        self.tgtNum = tgtNum
        self.labels = None
        return

    def generateOneImage(self):
        posList = []
        guard = self.tgtNum*10
        while len(posList)<self.tgtNum:
            guard -= 1
            if guard < 0:
                guard = self.tgtNum * 10
                posList = []
            pos = (np.random.randint(0,len(image)),
                np.random.randint(14,self.size[0]-14),
                np.random.randint(14,self.size[1]-14),0)
                #np.random.uniform(-30.,+30))
            isGood = True
            for a in posList:
                if np.sqrt( (pos[1]-a[1])**2 + (pos[2]-a[2])**2) < self.minLen: isGood=False
                #if pos[1]<28 or (self.size[0]-28)<pos[1] : isGood=False
                #if pos[2]<28 or (self.size[1]-28)<pos[2] : isGood=False
            if not isGood : continue
            posList.append(pos)

        res = Image.fromarray(np.zeros(self.size,dtype=np.float32))
        lab = []

        for a in posList:
            idx,x,y,r = a
            img  = Image.fromarray(image[idx]*255.)
            img  = img.rotate(r)
            res.paste(img,(int(x-14),int(y-14),int(x+14),int(y+14)))
            lab.append(label[idx])
        return np.asarray(res),lab

    def buildOneHot(self,x,f):
        res = np.zeros((self.tgtNum+1),dtype=np.int32)
        num = int(sum([f(a) for a in x]))
        res[num] = 1
        return res

    def prepareImages(self,fpath,nImages):
        if os.path.exists(fpath):
            shutil.rmtree(fpath)
        os.makedirs(fpath)

        fcsv = open(os.path.join(fpath,"labels.csv"),"w")
        labcsv = csv.writer(fcsv)
        for i in range(nImages):
            if i%1000==0: print i
            img,lab = self.generateOneImage()
            name = "%d.png"%i
            Image.fromarray(img).convert("RGB").save(os.path.join(fpath,name))
            line  = [name]
            line += lab
            labcsv.writerow(line)
            #labcsv.flush()
        fcsv.close()

    def loadBatch(self,fpath,nBatch,f):
        if not self.labels:
            modelCSVFile = open(os.path.join(fpath,"labels.csv"),"r")
            modelCSV = csv.reader(modelCSVFile)
            self.labels = []
            self.images = []
            for line in modelCSV:
                self.images.append(line[0])
                self.labels.append([int(x) for x in line[1:]])
                assert len(line[1:]) == self.tgtNum
            modelCSVFile.close()
        nLabels = len(self.labels)

        x = np.zeros((nBatch, self.size[0], self.size[1]), np.float32)
        t = np.zeros((nBatch, self.tgtNum+1), np.int32)

        for i in range(nBatch):
            ridx = random.randint(0,nLabels-1)
            img = np.asarray(Image.open(os.path.join(fpath,self.images[ridx])).convert("L"))
            lab =  self.buildOneHot(self.labels[ridx],f)
            x[i,:,:] = img
            t[i,:]   = lab

        return x,t

    def generateBatch(self,nBatch,f):
        x = np.zeros((nBatch, self.size[0], self.size[1]), np.float32)
        t = np.zeros((nBatch, self.tgtNum+1), np.int32)

        for i in range(nBatch):
            img,lab = self.generateOneImage()
            lab =  self.buildOneHot(lab,f)
            x[i,:,:] = img
            t[i,:]   = lab

        return x,t

class net:
    def __init__(self,nMax,nBatch,inputSize,learning_rate,saveFolder):
        self.nMax = nMax
        self.nBatch = nBatch
        self.inputSize = inputSize
        self.learning_rate = learning_rate
        self.saveFolder = saveFolder
        self.collection_loss = None
        self.collection_accuracy = None
        self.model()
        return

    def _fc_variable(self, weight_shape):
        input_channels  = int(weight_shape[0])
        output_channels = int(weight_shape[1])
        weight_shape = (input_channels, output_channels)
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        return weight, bias

    def _conv_variable(self, weight_shape):
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        input_channels  = int(weight_shape[2])
        output_channels = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        d = 1.0 / np.sqrt(input_channels * w * h)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def leakyReLU(self,x,alpha=0.1):
        return tf.maximum(x*alpha,x)

    def model(self):
        with tf.variable_scope("net") as scope:
            #############################
            ### input variable definition
            self.x  = tf.placeholder(tf.float32, [self.nBatch, self.inputSize[0], self.inputSize[1]],name="x")
            self.t  = tf.placeholder(tf.float32, [self.nBatch, self.nMax],name="t")

            # start
            h = self.x

            with tf.variable_scope("conv1"):
                # conv1
                h = tf.expand_dims(h,axis=3)
                self.conv1_w, self.conv1_b = self._conv_variable([15,15,1,10])
                h = self._conv2d(h, self.conv1_w, stride=1) + self.conv1_b
                h = self.leakyReLU(h)

                # maxpool1
                h = tf.nn.max_pool(h, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

            with tf.variable_scope("conv2"):
                # conv2
                self.conv2_w, self.conv2_b = self._conv_variable([3,3,10,10])
                h = self._conv2d(h, self.conv2_w, stride=1) + self.conv2_b
                h = self.leakyReLU(h)

                # maxpool2
                h = tf.nn.max_pool(h, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

            with tf.variable_scope("fc1"):
                # fc1
                sb,sh,sw,sf = [int(a) for a in h.get_shape()]
                print(sh,sb,sf)
                h = tf.reshape(h,[self.nBatch , sh*sw*sf])
                self.fc1_w, self.fc1_b = self._fc_variable([sh*sw*sf, 32])
                h = tf.matmul(h, self.fc1_w) + self.fc1_b
                h = self.leakyReLU(h)

            with tf.variable_scope("fc2"):
                # fc2
                self.fc2_w, self.fc2_b = self._fc_variable([32, self.nMax])
                h = tf.matmul(h, self.fc2_w) + self.fc2_b

            with tf.variable_scope("softmax"):
                self.y  = tf.nn.softmax(h)
            
            #############################
            ### loss definition
            #self.loss_class  = tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(self.y,[-1]),labels=tf.reshape(self.t,[-1]))
            self.loss_class  = -tf.reduce_sum( tf.reshape(self.t,[-1]) * tf.log(tf.reshape(self.y,[-1])) )

            self.loss_l2     = 1e-10 * (   tf.nn.l2_loss(self.conv1_w) + tf.nn.l2_loss(self.conv2_w)
                                         + tf.nn.l2_loss(self.fc1_w)   + tf.nn.l2_loss(self.fc2_w) )

            self.loss_total = self.loss_class+ self.loss_l2

            #############################
            ### accuracy definition
            self.accuracy = tf.reduce_mean(tf.cast( tf.equal(tf.argmax(self.y,1), tf.argmax(self.t,1)) , tf.float32))

            ### summary
            tf.summary.scalar("loss_total",self.loss_total)
            tf.summary.scalar("loss_class",self.loss_class)
            tf.summary.scalar("loss_l2"   ,self.loss_l2)
            tf.summary.scalar("accuracy"  ,self.accuracy)
            tf.summary.histogram("conv1_w"   ,self.conv1_w)
            tf.summary.histogram("conv1_b"   ,self.conv1_b)
            tf.summary.histogram("conv2_w"   ,self.conv2_w)
            tf.summary.histogram("conv2_b"   ,self.conv2_b)
            tf.summary.histogram("fc1_w"   ,self.fc1_w)
            tf.summary.histogram("fc1_b"   ,self.fc1_b)
            tf.summary.histogram("fc2_w"   ,self.fc2_w)
            tf.summary.histogram("fc2_b"   ,self.fc2_b)

            #############################
            ### optimizer
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer = optimizer.minimize(self.loss_total)

        #############################
        ### session
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.15))
        self.sess = tf.Session(config=config)

        #############################
        ### saver
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        if self.saveFolder: self.writer = tf.summary.FileWriter(self.saveFolder, self.sess.graph)

        #############################
        self.sess.run(tf.global_variables_initializer())

        return

    def train(self,step,batch_x,batch_t):
        _,summary,loss,accuracy = self.sess.run([self.optimizer,self.summary,self.loss_total,self.accuracy],feed_dict={self.x:batch_x,self.t:batch_t})
        if not self.collection_loss    : self.collection_loss     = collections.deque(maxlen=100)
        if not self.collection_accuracy: self.collection_accuracy = collections.deque(maxlen=100)
        self.collection_loss.append(loss)
        self.collection_accuracy.append(accuracy)
        if step%10==0:
            print("%5d: loss = %.3e  accuracy = %.2f%%"%(step,np.mean(self.collection_loss),np.mean(self.collection_accuracy)*100.))
        if step>0 and step%100==0:
            self.writer.add_summary(summary,step)
            self.saver.save(self.sess,os.path.join(self.saveFolder,"model.ckpt"),step)
        return

    def load(self, model_path=None):
        if model_path: self.saver.restore(self.sess, model_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgNMax",type=int,default=5)
    parser.add_argument("--nBatch",type=int,default=64)
    parser.add_argument("--inputSize",type=int,default=100)
    parser.add_argument("--learnRate",type=float,default=1e-4)
    parser.add_argument("--saveFolder",type=str,default="models")
    parser.add_argument("--reload",type=str,default=None)
    args = parser.parse_args()

    smp = sampleGen(size=[args.inputSize,args.inputSize],minLen=28,tgtNum=args.imgNMax)
    #smp.prepareImages("images",1000)
    #smp.prepareImages("images",100*1000)

    n = net(args.imgNMax+1,args.nBatch,[args.inputSize,args.inputSize],learning_rate=args.learnRate,saveFolder=args.saveFolder)
    n.load(args.reload)
    step = 0

    def f_odd(x):
        return x%2 # 1 in case of odd

    while True:
        batch_x, batch_t = smp.generateBatch(args.nBatch,f_odd)
        #batch_x, batch_t = smp.loadBatch("images",args.nBatch,f_odd)
        n.train(step,batch_x,batch_t)
        step += 1
