from torchvision.datasets import CIFAR100
import numpy as np
from PIL import Image


class iCIFAR100(CIFAR100):
    def __init__(self,root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False):
        super(iCIFAR100,self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        self.target_test_transform=target_test_transform
        self.test_transform=test_transform
        self.TrainData = None
        self.TrainLabels = None
        self.TestData = None
        self.TestLabels = None
        # self.TestData = []
        # self.TestLabels = []

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label
    
    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            if data.shape[0] == 0:
                continue  # 跳过没有样本的类别
            datas.append(data)
            labels.append(np.full((data.shape[0],), label))
        
        if len(datas) == 0:
            return  # 没有数据，不处理

        datas, labels = self.concatenate(datas, labels)

        if self.TestData is None:
            self.TestData = datas
            self.TestLabels = labels
        else:
            self.TestData = np.concatenate((self.TestData, datas), axis=0)
            self.TestLabels = np.concatenate((self.TestLabels, labels), axis=0)

        print("the size of test set is %s" % (str(self.TestData.shape)))

        print("the size of test label is %s" % (str(self.TestLabels.shape)))



    # def getTestData(self, classes):
    #     datas,labels=[],[]
    #     for label in range(classes[0], classes[1]):
    #         data = self.data[np.array(self.targets) == label]
    #         datas.append(data)
    #         labels.append(np.full((data.shape[0]), label))
    #     datas,labels=self.concatenate(datas,labels)
    #     self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
    #     self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
    #     print("the size of test set is %s"%(str(self.TestData.shape)))
    #     print("the size of test label is %s"%str(self.TestLabels.shape))

    def getTrainData(self, classes, exemplar_set):
        datas, labels = [], []

        # 如果已有 exemplar_set，则优先加入
        if len(exemplar_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length,), label) for label in range(len(exemplar_set))]

        # 加入当前任务类别的数据
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            if data.shape[0] == 0:
                continue  # 避免空类别
            datas.append(data)
            labels.append(np.full((data.shape[0],), label))

        if len(datas) == 0:
            return  # 没有数据，不处理

        datas, labels = self.concatenate(datas, labels)

        # 确保 TrainData 和 TrainLabels 是 ndarray
        if self.TrainData is None:
            self.TrainData = datas
            self.TrainLabels = labels
        else:
            self.TrainData = np.concatenate((self.TrainData, datas), axis=0)
            self.TrainLabels = np.concatenate((self.TrainLabels, labels), axis=0)

        print("the size of train set is %s" % (str(self.TrainData.shape)))
        print("the size of train label is %s" % (str(self.TrainLabels.shape)))

    # def getTrainData(self,classes,exemplar_set):

    #     datas,labels=[],[]  
    #     if len(exemplar_set)!=0:
    #         datas=[exemplar for exemplar in exemplar_set ]
    #         length=len(datas[0])
    #         labels=[np.full((length),label) for label in range(len(exemplar_set))]

    #     for label in range(classes[0],classes[1]):
    #         data=self.data[np.array(self.targets)==label]
    #         datas.append(data)
    #         labels.append(np.full((data.shape[0]),label))
    #     self.TrainData,self.TrainLabels=self.concatenate(datas,labels)
    #     print("the size of train set is %s"%(str(self.TrainData.shape)))
    #     print("the size of train label is %s"%str(self.TrainLabels.shape))

    def getTrainItem(self,index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]

        if self.transform:
            img=self.transform(img)

        if self.target_transform:
            target=self.target_transform(target)

        return index,img,target

    def getTestItem(self,index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img=self.test_transform(img)

        if self.target_test_transform:
            target=self.target_test_transform(target)

        return index, img, target

    def __getitem__(self, index):
        if self.TrainData is not None and len(self.TrainData) > 0:
            return self.getTrainItem(index)
        elif self.TestData is not None and len(self.TestData) > 0:
            return self.getTestItem(index)
        else:
            return 0


    def __len__(self):
        if self.TrainData is not None and len(self.TrainData) > 0:
            return len(self.TrainData)
        elif self.TestData is not None and len(self.TestData) > 0:
            return len(self.TestData)
        else:
            return 0

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label] #筛选所有类别为i标为label的图片


