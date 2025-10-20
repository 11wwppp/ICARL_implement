from shapely import length
from torchvision.datasets import VisionDataset
import os
import numpy as np
from PIL import Image
class CustomDataset(VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 test_transform=None, target_test_transform=None):
        super(CustomDataset, self).__init__(root, transform=transform,target_transform=target_transform)
        self.train = train
        self.test_transform = test_transform
        self.target_test_transform = target_test_transform
        
        self.date = [] # 存储图片数据
        self.targets = [] # 存储标签
        self.classes = []

        self._load_data()
    
    def _load_data(self):
        """加载自定义数据集"""
        if self.train:
            data_path = os.path.join(self.root,'train')
        else:
            data_path = os.path.join(self.root,'test')
        
        # 假设目录结构为: root/train_or_test/class_name/image.jpg
        class_names = sorted(os.listdir(data_path)) # 获取所有类别名称
        self.classes = class_names
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

        for class_name in class_names:
            class_path = os.path.join(data_path, class_name) # 获取某类文件夹路径
            if os.path.isdir(class_path):
                class_idx = class_to_idx[class_name] # 获取类别标签
                for img_name in os.listdir(class_path): # 遍历该类文件夹下的所有图片
                    if img_name.lower().endswith(('.png','.jpg','.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        self.data.append(img_path)
                        self.targets.append(class_idx)
        self.data = np.array(self.data)
        self.targets =np.array(self.targets)

    def get_image_class(self, label):
        """获取特定类别的所有图片"""
        indices = np.where(self.targets == label)[0] 
        image_paths = [self.data[i] for i in indices]
        return image_paths
    
    def getTrainData(self, classes, exemplar_set):
        '''获取训练数据，支持增量学习'''
        datas, labels = [], []

        # 如果已有exemplar_set 优先加入
        if exemplar_set is not None and len(exemplar_set) > 0:
            datas = [es for es in exemplar_set]
            if len(datas) > 0:
                length = len(datas[0])
                labels = [np.full((length,), label) for label in range(len(exemplar_set))]

        # 加入当然任务类别的数据
        for label in range(classes[0],classes[1]):
            image_paths = self.get_image_class(label)
            if len(image_paths) > 0:
                datas.append(image_paths)
                labels.append(np.full((len(image_paths),), label))
            
        if len(datas) == 0:
            self.TrainData = np.array([])
            self.TrainLabel = np.array([])
            return
        
        # 合并数据
        con_data = datas[0]
        con_label = labels[0]

        for i in range(1,len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis = 0)
            con_label = np.concatenate((con_label, labels[i]), axis = 0)

        self.TrainData = con_data
        self.TrainLabel = con_label

        print("the size of train set is %s" % (str(len(self.TrainData))))
        print("the size of train label is %s" % (str(len(self.TrainLabel))))

    def getTestData(self, classes):
        '''获取测试数据，支持增量学习'''
        datas, labels = [], []

        # 加入当前任务类别的数据
        for label in range(classes[0],classes[1]):
            image_paths = self.get_image_class(label)
            if len(image_paths) > 0:
                datas.append(image_paths)
                labels.append(np.full((len(image_paths),), label))
            
        if len(datas) == 0:
            self.TrainData = np.array([])
            self.TrainLabel = np.array([])
            return
        
        # 合并数据
        con_data = datas[0]
        con_label = labels[0]

        for i in range(1,len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis = 0)
            con_label = np.concatenate((con_label, labels[i]), axis = 0)

        if hasattr(self, 'TestData') and len(self.TestData) > 0:
            self.TestData = np.concatenate((self.TestData, con_data), axis = 0)
            self.TestLabel = np.concatenate((self.TestLabel, con_label), axis = 0)
        else:   
            self.TestData = con_data
            self.TestLabel = con_label

        print("the size of test set is %s" % (str(len(self.TestData))))
        print("the size of test label is %s" % (str(len(self.TestLabel))))

    def __getitem__(self, index):
        if hasattr(self, 'TrainData') and len(self.TrainData) >0: #检查当前对象是否包含TrainData属性
            return  self.getTrainData(index)
        elif hasattr(self, 'TestData') and len(self.TestData) >0:
            return self.getTestData(index)
        else:
            img_path = self.data[index]
            target =self.targets[index]
            img = Image.open(img_path).convert('RGB')
            if self.train and self.transform:
                img = self.transform(img)
            elif not self.train and self.test_transform:
                img = self.test_transform(img)
            
            return index, img, target

    def __len__(self):
        if hasattr(self,'TrainData') and len(self.TrainData) > 0:
            return len(self.TrainData)
        elif hasattr(self,'TestData') and len(self.TestData) > 0:
            return len(self.TestData)
        else:
            return len(self.data)