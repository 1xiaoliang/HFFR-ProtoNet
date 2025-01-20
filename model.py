import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features =  models.resnet101(weights=models.ResNet101_Weights)
        # self.features.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def forward(self, x):
        X=self.features.maxpool(self.features.relu(self.features.bn1(self.features.conv1(x))))
        # 高中低特征
        Shallow_feature=self.features.layer2(self.features.layer1(X))
        Common_feature=self.features.layer3(Shallow_feature)
        Deep_feature=self.features.layer4(Common_feature)
        # 取均值
        B=Shallow_feature.shape[0]
        Shallow_feature=Shallow_feature.reshape(B,Shallow_feature.shape[1]*Shallow_feature.shape[2],-1).mean(-1)
        Common_feature=Common_feature.reshape(B,Common_feature.shape[1]*Common_feature.shape[2],-1).mean(-1)
        Deep_feature=Deep_feature.reshape(B,Deep_feature.shape[1]*Deep_feature.shape[2],-1).mean(-1)
        return Shallow_feature,Common_feature,Deep_feature  # 输出尺寸 [batch_size, 512, H/32, W/32]


class LocalAttentionModule(nn.Module):
    def __init__(self, channels=3):
        super(LocalAttentionModule, self).__init__()
        # 使用转置卷积来生成与输入图像尺寸一致的注意力图
        self.attention_conv = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        self.deconv = nn.ConvTranspose2d(3, 3, kernel_size=3, padding=1)

        self.sigmoid=nn.Sigmoid()


    def forward(self, x):
       # 计算注意力图
        attention_map = self.attention_conv(x)
        attention_map = self.deconv(attention_map)  # 放大到与输入图像相同尺寸
        attention_weights = self.sigmoid(attention_map)  # 归一化到 [0, 1] 区间
        
        # 将注意力权重应用于原始特征图
        # attention_weights = torch.where(attention_weights < 0.5, torch.tensor(0.0, device=attention_weights.device), attention_weights) # 非前景部分mask
        attended_features = x * attention_weights#.expand_as(x) #+ x  # Residual connection
        
        return attended_features, torch.mean(attention_weights,dim=1,keepdim=True)
class FeatureSelect(nn.Module):
    def __init__(self):
        super().__init__()
        self.pro_layer=nn.Linear(16384,8192)
        self.atten=nn.TransformerEncoderLayer(d_model=8192,nhead=16,batch_first=True) #8192
        self.downsampe=nn.Sequential(
            nn.Linear(8192,4096),
            nn.ReLU(),
            nn.Linear(4096,2048),
            nn.ReLU()
        )
    def forward(self,Shallow_feature,Common_feature,Deep_feature):
        Shallow_feature=self.pro_layer(Shallow_feature)
        Common_feature=self.pro_layer(Common_feature)
        Deep_feature=self.pro_layer(Deep_feature)

        fuson_feature=torch.stack((Shallow_feature,Common_feature,Deep_feature),dim=-1).transpose(1,2)
        fuson_feature=self.atten.forward(fuson_feature)
        
        fuson_feature=torch.mean(fuson_feature,dim=1).squeeze() 
        fuson_feature=self.downsampe(fuson_feature)
        return fuson_feature

# 主模型
class HFFR_ProtoNet(nn.Module):
    def __init__(self, num_class=200):
        super(HFFR_ProtoNet, self).__init__()
        self.local_attention = LocalAttentionModule(channels=3)
        self.feature_extractor=FeatureExtractor()
        self.AdaptiveChoose=FeatureSelect()
        self.classify = nn.Linear(2048,num_class)
        # self.classify = nn.Linear(16384,num_class)
    def forward(self, x):
        # 特征增强,以区分前景后景
        attended_features, attention_weights = self.local_attention(x)
        # 高中低维度特征提取
        Shallow_feature,Common_feature,Deep_feature = self.feature_extractor(attended_features)
        # 自适应特征选择
        fusonFeature=self.s(Shallow_feature,Common_feature,Deep_feature)
        # fusonFeature=Deep_feature
        # 分类头
        output = self.classify(fusonFeature)
        
        return output,fusonFeature,attention_weights

if __name__ == '__main__':
    model = HFFR_ProtoNet()

    input_tensor = torch.randn(2, 3, 256, 256)  # 示例输入张量
    with torch.no_grad():
        output,fusonFeature,attention_weights = model(input_tensor)
        print("output shape:", output.shape)  
        print("fusonFeature shape:", fusonFeature.shape)  
        print("attention_weights shape:", attention_weights.shape)  