from typing import OrderedDict
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import HFFR_ProtoNet
import torchvision.transforms as transforms
from dataset import get_task
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter 
import numpy as np
import copy
from tqdm import tqdm
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

def compute_cosine_loss(query_features, prototypes, labels):
    """
    query_features: [batch_size, feature_dim] 查询样本的特征向量
    prototypes: [n_way, feature_dim] 每个类别的原型
    labels: [batch_size] 真实标签（已映射）
    """
    # 计算查询样本与每个类别原型之间的余弦相似度
    # similarities = F.cosine_similarity(query_features.unsqueeze(1), prototypes, dim=-1)#.softmax(-1)

    distances = torch.cdist(query_features, prototypes)  # 计算查询样本与原型之间的欧几里得距离
    distances=distances/distances.max()
    similarities = -distances*10  # 计算负距离  通过缩放因子放大负距离
    # 使用 softmax 对负距离进行平滑，以计算相似度分布
    similarities = F.softmax(similarities, dim=-1)
    
    # 使用相似度作为输入给交叉熵损失函数
    return F.cross_entropy(similarities, labels)
def evaluate_model(model, df_test: pd.DataFrame, n_way, k_shot, query_size, transform, writer, episode, num_episodes=600,device=torch.device('cuda:0')):
    model.eval()  
    accuracies = []

    with torch.no_grad():
        for i in tqdm(range(num_episodes), desc='Evaluating',ncols=80):
            # 获取任务（测试）
            support_loader, query_loader = get_task(df_test, n_way, k_shot, query_size, transform=transform)

            # 计算支持集的类别原型
            support_images, support_labels = next(iter(support_loader))
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            _, support_features, _ = model(support_images)  # 使用原始模型而不是快参模型
            
            mapped_support_labels = map_labels_to_indices(support_labels, n_way).to(device)
            prototypes = compute_prototypes(support_features, mapped_support_labels, n_way)

            # 在查询集上评估性能
            correct = 0
            total = 0
            for batch in query_loader:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                _, query_features, _ = model(images)  # 获取查询样本的特征
                
                # 计算查询样本与类别原型之间的余弦相似度
                # similarities = F.cosine_similarity(query_features.unsqueeze(1), prototypes, dim=-1).softmax(-1)
                distances = torch.cdist(query_features, prototypes)  # 使用欧几里得距离
                similarities = (-distances).softmax(-1)
                
                # 根据相似度预测标签
                predicted_labels = torch.argmax(similarities, dim=1)
                
                # 将原始标签映射为相对索引
                mapped_query_labels = map_labels_to_indices(labels, n_way).to(device)
                
                # 统计正确预测的数量
                correct += (predicted_labels == mapped_query_labels).sum().item()
                total += len(labels)

            accuracy = correct / total
            accuracies.append(accuracy)

            # if writer and episode is not None:
            #     writer.add_scalar('Accuracy/test', accuracy, episode * num_episodes + i)

    mean_accuracy = sum(accuracies) / len(accuracies)
    print(f'Mean Accuracy over {num_episodes} episodes: {mean_accuracy:.4f}')
    writer.add_scalar('Accuracy/test', mean_accuracy, episode)
    
    return mean_accuracy
def metaTrain(model, optimizer, num_episodes, n_way, k_shot, query_size, inner_loop_lr, inner_loop_steps, transform, train_df, eval_df, device,scheduler):
    writer = SummaryWriter(log_dir='/home/liangxiaoyuan/小样本学习/logs/metatrain/now')  
    model = model.to(device)
    best_acc=0
    for episode in range(num_episodes):
        episode_loss=[]
        # 获取任务（训练）
        support_loader, query_loader = get_task(train_df, n_way, k_shot, query_size, transform=transform)

        # 创建模型的深度拷贝用于保存快参
        fast_model = copy.deepcopy(model)
        fast_model.to(device)
        fast_optimizer = Adam(fast_model.parameters(), lr=inner_loop_lr)
        
        for batch in support_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # 内循环：使用支持集更新快参
            for _ in range(inner_loop_steps):
                fast_optimizer.zero_grad()
                # 使用当前快参进行前向传播
                _, features, _ = fast_model(images) 
                # 将标签映射为相对索引
                mapped_labels = map_labels_to_indices(labels, n_way).to(device)
                # 计算类别原型
                prototypes = compute_prototypes(features, mapped_labels, n_way)
                # 计算损失并计算梯度
                loss = compute_cosine_loss(features, prototypes, mapped_labels)
                loss.backward()  # 更新快参
                fast_optimizer.step()
               
               

            # model.load_state_dict(fast_model.state_dict()) # 这里是否需要保留
            # 外循环：在查询集上评估性能并更新慢参
            query_images, query_labels = next(iter(query_loader))
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            _, query_features, _ = fast_model(query_images)  # 获取查询样本的特征
            
            # 计算类别原型
            support_images, support_labels = next(iter(support_loader))
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            _, support_features, _ = fast_model(support_images)  # 获取支持样本的特征 是不是要用fast_model
            
            # 将支持集标签映射为相对索引
            mapped_support_labels = map_labels_to_indices(support_labels, n_way).to(device)
            prototypes = compute_prototypes(support_features, mapped_support_labels, n_way)

            # 将查询集标签映射为相对索引
            mapped_query_labels = map_labels_to_indices(query_labels, n_way).to(device)

            # 计算查询集上的损失
            query_loss = compute_cosine_loss(query_features, prototypes, mapped_query_labels)
            episode_loss.append(query_loss.item())
            # 更新慢参
            optimizer.zero_grad()
            query_loss.backward()
            optimizer.step()
        scheduler.step()
        writer.add_scalar('Loss/train', sum(episode_loss)/len(episode_loss), episode) 
        if episode % 50==0:
            print(f'Episode [{episode}/{num_episodes}], Loss: {sum(episode_loss)/len(episode_loss):.4f}')
        # 每隔一定次数episode进行验证评估
        if episode % 200 == 0 and episode > 0:
            mean_accuracy= evaluate_model(model, eval_df, n_way, k_shot, query_size, transform, writer, episode,num_episodes=600, device=device)
            # 保存模型
            if mean_accuracy>best_acc:
                best_acc=mean_accuracy
                torch.save(model.state_dict(), f'/home/liangxiaoyuan/小样本学习/logs/HFFR_ProtoNet.pth')
    torch.save(model.state_dict(), f'/home/liangxiaoyuan/小样本学习/logs/HFFR_ProtoNet-last.pth')



if __name__=='__main__':
    # 元训练配置
    num_episodes = 500000
    n_way = 5
    k_shot = 1
    query_size = 4
    inner_loop_lr = 1e-3
    inner_loop_steps = 2  # 内循环更新次数

    model = HFFR_ProtoNet(num_class=n_way)
    print(model.load_state_dict(torch.load('/home/liangxiaoyuan/小样本学习/logs/pretrain_CUB/best_pretrain.pt')))
    # print(model.load_state_dict(torch.load('/home/liangxiaoyuan/小样本学习/logs/HFFR_ProtoNet.pth')))

    for na,par in model.named_parameters():
        if  ('feature_extractor' in na and 'layer4' not in na):
            par.requires_grad=False
        if par.requires_grad:
            print(na)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # 数据预处理变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 分离元学习训练集和验证集
    df = pd.read_csv('/home/liangxiaoyuan/小样本学习/data/CUB_200_2011.csv')  
    train_df = df[df['set'] == 'meta_train']
    train_df=train_df.reset_index(drop=True)
    eval_df = df[df['set'] == 'test']
    eval_df=eval_df.reset_index(drop=True)


    device=torch.device('cuda:0')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_episodes, eta_min=0)
    
    metaTrain(model, optimizer, num_episodes, n_way, k_shot, query_size, inner_loop_lr, inner_loop_steps, transform, train_df, eval_df,device,scheduler)

    