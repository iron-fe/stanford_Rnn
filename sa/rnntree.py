import numpy as np

class rnntree:
    def __init__(self,d,sl,words_embedded):
        """
        d是词向量的维数
        s1可能是词向量的个数
        word_embdded是词向量
        pp和kids记录了树的信息
        """
        self.pp = np.zeros([(2*sl-1),1])
        self.nodeScores = np.zeros([2*sl-1,1])#这个有可能是情感误差
        self.nodeNames = range(2*sl-1)
        self.kids = np.zeros([2*sl-1,2]).astype(int)
        self.numkids = np.ones([2*sl-1,1]).astype(int)
        self.node_y1c1 = np.zeros([d,2*sl-1])
        self.node_y2c2 = np.zeros([d,2*sl-1])
        self.freq = np.zeros([2*sl-1,1])
        self.nodeFeatures = np.hstack([words_embedded,np.zeros([d, sl-1])])  #nodeFeatures存的是p，得分
        self.nodeFeatures_unnormalized = np.hstack([words_embedded,np.zeros([d, sl-1])]) # 标准化的p
        self.nodeDelta_out1 = np.zeros([d,2*sl-1])
        self.nodeDelta_out2 = np.zeros([d,2*sl-1])
        self.parentDelta = np.zeros([d,2*sl-1])
