from dataset import ImageDataset
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from model import HFFR_ProtoNet
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
logs=SummaryWriter('/home/liangxiaoyuan/小样本学习/logs/pretrain')

def train(epoches,model,traindata,evaldata,loss_func,opt,device,lr_drop):
    Th=10
    flag=0
    global_step,best_acc=0,0
    model=model.to(device)
    for epoch in range(epoches):

        if epoch>Th and flag==0:
            for na,par in model.named_parameters():
                par.requires_grad=True
            flag=1
        if epoch>=Th:
            lr_drop.step()
        model.train()
        train_loss=0
        for img,label in tqdm(traindata,desc='training',ncols=80):
            global_step+=1
            img=img.to(device)
            label=label.to(device)
            out,_,attention_weights=model(img)
            loss=loss_func(out,label)
            logs.add_scalar(tag='train_loss_step',scalar_value=loss.item(),global_step=global_step)

            train_loss+=loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()
        logs.add_scalar(tag='train_loss',scalar_value=train_loss,global_step=epoch)
        model.eval()
        pred_L=[]
        label_L=[]
        with torch.no_grad():
            for img,label  in tqdm(evaldata,desc='evaling',ncols=80):
                img=img.to(device)
                out,_,_=model(img)
                pred=torch.argmax(out,dim=-1).cpu().tolist()
                pred_L+=pred
                label=label.tolist()
                label_L+=label
        accuracy,f1,precision,recall=accuracy_score(label_L,pred_L),f1_score(label_L,pred_L,average='weighted'),precision_score(label_L,pred_L,average='weighted'),recall_score(label_L,pred_L,average='weighted')
        logs.add_scalar(tag='accuracy',scalar_value=accuracy,global_step=epoch)
        logs.add_scalar(tag='f1',scalar_value=f1,global_step=epoch)
        logs.add_scalar(tag='precision',scalar_value=precision,global_step=epoch)
        logs.add_scalar(tag='recall',scalar_value=recall,global_step=epoch)
        if accuracy>best_acc:
            best_acc=accuracy
            torch.save(model.state_dict(),'/home/liangxiaoyuan/小样本学习/logs/best_pretrain.pt')
        print(f"{epoch}/{epoches} accuracy={accuracy} precision={precision} recall={recall}")
    return model

if __name__=='__main__':
    epoches=1000
    model=HFFR_ProtoNet(num_class=120)
    # model.load_state_dict(torch.load('/home/liangxiaoyuan/小样本学习/logs/best_pretrain.pt'))
    # 正常用法
    df=pd.read_csv('/home/liangxiaoyuan/小样本学习/data/mini_imagenet.csv')
    # df=pd.read_csv('/home/liangxiaoyuan/小样本学习/data/CUB_200_2011.csv')
    df=df[df['set']=='train']
    train_csv = pd.DataFrame(columns=df.columns)
    eval_csv = pd.DataFrame(columns=df.columns)

    # 按每个label划分训练集和验证集
    for label in df['label'].unique():
        # 筛选出当前标签的数据
        label_data = df[df['label'] == label]
        
        # 使用 train_test_split 按照 10% 划分验证集
        trainD, valD = train_test_split(label_data, test_size=0.1, random_state=42)
        
        # 将划分后的数据添加到训练集和验证集中
        train_csv = pd.concat([train_csv, trainD])
        eval_csv = pd.concat([eval_csv, valD])

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    dataset=ImageDataset(train_csv, transform=transform)
    traindata= DataLoader(dataset, batch_size=32,shuffle=True,drop_last=True)

    dataset=ImageDataset(eval_csv, transform=transform)
    evaldata= DataLoader(dataset, batch_size=32)

    loss_func=torch.nn.CrossEntropyLoss()
    opt=torch.optim.RMSprop(model.parameters(),lr=1e-5)
    device=torch.device('cuda:0')

    lr_drop=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epoches, eta_min=0)

    for na,par in model.named_parameters():
        if 'feature_extractor' in na:
            print(na)
            par.requires_grad=False
    model=train(epoches,model,traindata,evaldata,loss_func,opt,device,lr_drop)

    