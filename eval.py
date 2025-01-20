import torchvision.transforms as transforms
import pandas as pd
import torch
from model import HFFR_ProtoNet
from dataset import get_task
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import sem
import numpy as np
def compute_prototypes(support_features, support_labels, n_way):
    """
    计算每个类别的原型。
    Args:
        support_features: [num_samples, feature_dim] 支持集样本的特征向量
        support_labels: [num_samples] 支持集样本的真实标签
        n_way: 类别数量
        
    Returns:
        prototypes: [n_way, feature_dim] 每个类别的原型
    """
    prototypes = []
    for class_idx in range(n_way):
        class_features = support_features[support_labels == class_idx]
        prototype = class_features.mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes)
def map_labels_to_indices(labels,n_way=None):
    """
    将实际类别标签映射为相对索引 (0 到 n_way-1)。
    
    Args:
        labels: [num_samples] 实际类别标签
    Returns:
        mapped_labels: [num_samples] 映射后的相对索引标签
    """
    unique_labels = torch.unique(labels)
    label_to_index = {label.item(): idx for idx, label in enumerate(unique_labels)}
    mapped_labels = torch.tensor([label_to_index[label.item()] for label in labels], dtype=torch.long)
    return mapped_labels


def evaluate_model(model, df_test: pd.DataFrame, n_way, k_shot, query_size, transform, num_episodes=600, device=torch.device('cuda:0')):
    model=model.to(device)
    model.eval()  
    accuracies = []

    with torch.no_grad():
        for i in tqdm(range(num_episodes), desc='Evaluating',ncols=100):
            # 获取任务（测试）
            support_loader, query_loader = get_task(df_test, n_way, k_shot, query_size, transform=transform)

            # 计算支持集的类别原型
            support_images, support_labels = next(iter(support_loader))
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            _, support_features, _ = model(support_images)  
            
            mapped_support_labels = map_labels_to_indices(support_labels, n_way).to(device)
            prototypes = compute_prototypes(support_features, mapped_support_labels, n_way)

            # 在查询集上评估性能
            correct = 0
            total = 0
            for batch in query_loader:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                _, query_features, atten_weights = model(images)  # 获取查询样本的特征
                # torch.save(atten_weights,'/home/liangxiaoyuan/小样本学习/atten_weights.pt')
                # torch.save(images,'/home/liangxiaoyuan/小样本学习/images.pt')
                # dasdsa
                
                # 计算查询样本与类别原型之间的余弦相似度
                similarities = F.cosine_similarity(query_features.unsqueeze(1), prototypes, dim=-1).softmax(-1)
                # distances = torch.cdist(query_features, prototypes)  # 使用欧几里得距离
                # similarities = (-distances).softmax(-1)
                
                # 根据相似度预测标签
                predicted_labels = torch.argmax(similarities, dim=1)
                
                # 将原始标签映射为相对索引
                mapped_query_labels = map_labels_to_indices(labels, n_way).to(device)
                
                # 统计正确预测的数量
                correct += (predicted_labels == mapped_query_labels).sum().item()
                total += len(labels)

            accuracy = correct / total
            accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    std_error = sem(accuracies)  # 计算标准误差
    result_str = f'{mean_accuracy * 100:.2f}±{std_error * 100:.2f}'  
    print(f'Mean Accuracy over {num_episodes} episodes: {result_str}')
    
    return result_str


if __name__=='__main__':
    n_way=5
    k_shot=1
    query_size=8

    model = HFFR_ProtoNet(num_class=120)
    # print(model.load_state_dict(torch.load('/home/liangxiaoyuan/小样本学习/logs/HFFR_ProtoNet.pth')))
    print(model.load_state_dict(torch.load('/home/liangxiaoyuan/小样本学习/logs/best_pretrain.pt'),strict=False))
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    df = pd.read_csv('/home/liangxiaoyuan/小样本学习/data/mini_imagenet.csv')  
    # df = pd.read_csv('/home/liangxiaoyuan/小样本学习/data/CUB_200_2011.csv')  
    df_test = df[df['set'] == 'test']
    df_test=df_test.reset_index(drop=True)
    evaluate_model(model, df_test, n_way=n_way, k_shot=k_shot, query_size=query_size, transform=test_transform,num_episodes=600)