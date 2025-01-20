import pandas as pd
import random
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.gmm = GaussianMixture(n_components=2, random_state=42, max_iter=1)
    def __len__(self):
        return len(self.df)
    def get_Kmeans(self,image):
        # 获取图像的形状
        C, H, W = image.shape  
        # 将图像从 (B, C, H, W) 转换为 (B*H*W, C)
        pixels = image.permute(1, 2, 0).reshape(-1, C).cpu().numpy()  # 转为 NumPy 数组
        # 用 GaussianMixture 对每张图像进行聚类
        gmm_labels = self.gmm.fit_predict(pixels) # 初始化聚类标签
        segmented_img = gmm_labels.reshape(H, W)
        segmented_img+=1
        # 将所有分割结果堆叠成一个批次输出
        return torch.tensor(segmented_img)[None]
    def __getitem__(self, idx):
        
        try:
            img_path = self.df.iloc[idx]['img_path']
            label = self.df.iloc[idx]['label']
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(e)
            return self.__getitem__(idx+1)
        if self.transform:
            image = self.transform(image)
        # Kmeans_mask=self.get_Kmeans(image)
        # image=torch.concat((image,Kmeans_mask),dim=0)
        return image, label

def get_task(df: pd.DataFrame, n_way, k_shot, query_size, transform=None):
    """
    从给定的 DataFrame 中获取一个 n-way k-shot 任务。
    
    参数:
        df (pd.DataFrame): 包含 'img_path', 'label', 'label_name' 列的数据框。
        n_way (int): 类别数量。
        k_shot (int): 每个类别的样本数量（支持集）。
        query_size (int): 查询集中每个类别的样本数量。
        transform (callable, optional): 可选的图像变换函数。
        
    返回:
        tuple: 包含两个 DataLoader 对象 (support_loader, query_loader)。
    """
    # 获取所有类别的列表并检查样本数量
    all_classes = df['label'].unique()
    for cls in all_classes:
        if len(df[df['label'] == cls]) < k_shot + query_size:
            raise ValueError(f"Class {cls} has insufficient samples for the given k_shot and query_size.")

    # 随机选择 n_way 个类别
    selected_classes = random.sample(list(all_classes), n_way)

    support_samples = []
    query_samples = []

    for cls in selected_classes:
        # 获取属于该类的所有索引
        class_df = df[df['label'] == cls]
        class_indices = class_df.index.tolist()
        
        # 从中随机选择 k_shot + query_size 个样本
        selected_indices = random.sample(class_indices, k_shot + query_size)
        
        # 分割成支持集和查询集
        support_samples.extend(selected_indices[:k_shot])
        query_samples.extend(selected_indices[k_shot:])
    
    # 创建子集采样器
    support_sampler = SubsetRandomSampler(support_samples)
    query_sampler = SubsetRandomSampler(query_samples)

    # 创建数据集实例
    dataset = ImageDataset(df, transform=transform)

    # 创建数据加载器
    support_loader = DataLoader(dataset, batch_size=k_shot*n_way, sampler=support_sampler)
    query_loader = DataLoader(dataset, batch_size=query_size*n_way, sampler=query_sampler)

    return support_loader, query_loader
if __name__=='__main__':    
    # 正常用法
    # df=pd.read_csv('/home/liangxiaoyuan/project/小样本学习/CUB_200_2011.csv')
    # transform = transforms.Compose([transforms.Resize((84, 84)), transforms.ToTensor()])
    # dataset=ImageDataset(df, transform=transform)
    # loader= DataLoader(dataset, batch_size=12)
    # n-way n-shot用法
    df=pd.read_csv('/home/liangxiaoyuan/小样本学习/data/CUB_200_2011.csv')
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    support_loader, query_loader = get_task(df, n_way=5, k_shot=5, query_size=15, transform=transform)

    for x,y in support_loader:
        print(x.shape,y.shape)

    for x,y in query_loader:
        print(x.shape,y.shape)
    print(x[0])
    
