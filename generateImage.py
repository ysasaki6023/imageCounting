import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os,argparse
import collections

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
        return

    def generateOneImage(self):
        posList = []
        while len(posList)<(self.tgtNum-1):
            pos = (np.random.randint(0,len(image)),
                np.random.randint(0,self.size[0]),
                np.random.randint(0,self.size[1]),0)
                #np.random.uniform(-30.,+30))
            isGood = True
            for a in posList:
                if np.sqrt( (pos[1]-a[1])**2 + (pos[2]-a[2])**2) < self.minLen: isGood=False
                if pos[1]<28 or (self.size[0]-28)<pos[1] : isGood=False
                if pos[2]<28 or (self.size[1]-28)<pos[2] : isGood=False
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
        res = np.zeros((self.tgtNum),dtype=np.int32)
        num = int(sum([f(a) for a in x]))
        res[num] = 1
        return res

    def generateBatch(self,nBatch,f):
        x = np.zeros((nBatch, self.size[0], self.size[1]), np.float32)
        t = np.zeros((nBatch, self.tgtNum), np.int32)

        for i in range(nBatch):
            img,lab = self.generateOneImage()
            lab =  self.buildOneHot(lab,f)
            x[i,:,:] = img
            t[i,:]   = lab
            #print(lab)
            #Image.fromarray(img).show()
            #raw_input()

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
        #print(x.get_shape(),W.get_shape())
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

            # conv1
            h = tf.expand_dims(h,axis=3)
            self.conv1_w, self.conv1_b = self._conv_variable([3,3,1,32])
            h = self._conv2d(h, self.conv1_w, stride=1) + self.conv1_b
            h = self.leakyReLU(h)

            # conv2
            self.conv2_w, self.conv2_b = self._conv_variable([3,3,32,64])
            h = self._conv2d(h, self.conv2_w, stride=1) + self.conv2_b
            h = self.leakyReLU(h)

            # maxpool1
            h = tf.nn.max_pool(h, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

            # conv3
            self.conv3_w, self.conv3_b = self._conv_variable([3,3,64,64])
            h = self._conv2d(h, self.conv3_w, stride=1) + self.conv3_b
            h = self.leakyReLU(h)

            # conv4
            self.conv4_w, self.conv4_b = self._conv_variable([3,3,64,64])
            h = self._conv2d(h, self.conv4_w, stride=1) + self.conv4_b
            h = self.leakyReLU(h)

            # maxpool2
            h = tf.nn.max_pool(h, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

            # fc1
            sb,sh,sw,sf = [int(a) for a in h.get_shape()]
            print(sh,sb,sf)
            h = tf.reshape(h,[self.nBatch , sh*sw*sf])
            self.fc1_w, self.fc1_b = self._fc_variable([sh*sw*sf, 64])
            h = tf.matmul(h, self.fc1_w) + self.fc1_b
            h = self.leakyReLU(h)

            # fc2
            self.fc2_w, self.fc2_b = self._fc_variable([64, self.nMax])
            h = tf.matmul(h, self.fc2_w) + self.fc2_b

            self.y  = tf.nn.softmax(h)
            
            #############################
            ### loss definition
            self.loss_class  = tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(self.y,[-1]),labels=tf.reshape(self.t,[-1]))

            self.loss_l2     = 1e-10 * (   tf.nn.l2_loss(self.conv1_w) + tf.nn.l2_loss(self.conv2_w) + tf.nn.l2_loss(self.conv3_w) + tf.nn.l2_loss(self.conv4_w)
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
            tf.summary.histogram("conv3_w"   ,self.conv3_w)
            tf.summary.histogram("conv3_b"   ,self.conv3_b)
            tf.summary.histogram("conv4_w"   ,self.conv4_w)
            tf.summary.histogram("conv4_b"   ,self.conv4_b)
            tf.summary.histogram("fc1_w"   ,self.fc1_w)
            tf.summary.histogram("fc1_b"   ,self.fc1_b)
            tf.summary.histogram("fc2_w"   ,self.fc2_w)
            tf.summary.histogram("fc2_b"   ,self.fc2_b)

            #############################
            ### optimizer
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer = optimizer.minimize(self.loss_total)

        #############################
        ### saver
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        if self.saveFolder: self.writer = tf.summary.FileWriter(self.saveFolder)

        #############################
        ### session
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.15))
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        return

    def train(self,step,batch_x,batch_t):
        _,summary,loss,accuracy = self.sess.run([self.optimizer,self.summary,self.loss_total,self.accuracy],feed_dict={self.x:batch_x,self.t:batch_t})
        if not self.collection_loss    : self.collection_loss     = collections.deque(maxlen=100)
        if not self.collection_accuracy: self.collection_accuracy = collections.deque(maxlen=100)
        self.collection_loss.append(loss)
        self.collection_accuracy.append(accuracy)
        if step%100==0:
            print("%5d: loss = %.1e  accuracy = %.2f%%"%(step,np.mean(self.collection_loss),np.mean(self.collection_accuracy)*100.))
        if step>0 and step%100==0:
            self.writer.add_summary(summary,step)
            self.saver.save(self.sess,os.path.join(self.saveFolder,"model.ckpt"),step)
        return

    def load(self, model_path=None):
        if model_path: self.saver.restore(self.sess, model_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_max",type=int,default=7)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--input_size",type=int,default=150)
    parser.add_argument("--learn_rate",type=float,default=1e-4)
    parser.add_argument("--save_folder",type=str,default="models")
    parser.add_argument("--reload",type=str,default=None)
    args = parser.parse_args()

    nMax = args.n_max
    nBatch = args.batch_size
    inputSize = (args.input_size,args.input_size)
    n = net(nMax,nBatch,inputSize,learning_rate=args.learn_rate,saveFolder=args.save_folder)
    n.load(args.reload)
    step = 0
    smp = sampleGen(size=inputSize,minLen=28,tgtNum=nMax)

    def f_odd(x):
        return x%2 # 1 in case of odd

    while True:
        batch_x, batch_t = smp.generateBatch(nBatch,f_odd)
        n.train(step,batch_x,batch_t)
        step += 1
