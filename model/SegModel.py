import featurefusion_network as fusion
# import featureExtraction
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import unet
import unet_parts as unetpart
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DoubleConv(nn.Module):

    def __init__(self,inputChannel,outputChannel,midChannel=None):
        super().__init__()

        if not midChannel:
            midChannel = outputChannel
        self.double_conv=nn.Sequential(
            nn.Conv2d(in_channels=inputChannel,out_channels=midChannel,kernel_size=3,padding=1),
            nn.InstanceNorm2d(midChannel),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            nn.Conv2d(midChannel,outputChannel,kernel_size=3,padding=1),
            nn.InstanceNorm2d(outputChannel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    def forward(self,x):
        return self.double_conv(x)

#expand feature map to its original resolution
class expandLayer(nn.Module):
    def __init__(self,inputChannel,outputChannel=None):
        super().__init__()
        if not outputChannel:
            outputChannel = inputChannel
        self.expa=nn.Sequential(nn.ConvTranspose2d(inputChannel,outputChannel, kernel_size=2, stride=2),DoubleConv(outputChannel, outputChannel),
                                nn.ConvTranspose2d(outputChannel,outputChannel, kernel_size=2, stride=2),DoubleConv(outputChannel, outputChannel),
                                nn.ConvTranspose2d(outputChannel,outputChannel, kernel_size=2, stride=2),DoubleConv(outputChannel, outputChannel))
    def forward(self,x):
        return self.expa(x)


#feature fusion layer
class featureFusionLayer(nn.Module):
    def __init__(self,inputdim):
        super().__init__()
        self.ecaMag= fusion.ECA(dim=inputdim,heads=8,qkv_bias=False,qk_scale=None,dropout_rate=0.2)
        self.ecaVel= fusion.ECA(dim=inputdim,heads=8,qkv_bias=False,qk_scale=None,dropout_rate=0.2)

        self.cfa1 =fusion.CFA(dim=inputdim,hidden_dim=256,heads=8,qkv_bias=False,qk_scale=None,dropout_rate=0.2)
        self.cfa2 =fusion.CFA(dim=inputdim,hidden_dim=256,heads=8,qkv_bias=False,qk_scale=None,dropout_rate=0.2)
    def forward(self,mag,vel):
        magFeature=self.cfa1(mag,vel)
        velFeature=self.cfa2(vel,mag)

        magFeature=self.ecaMag(magFeature)
        velFeature=self.ecaVel(velFeature)
        return magFeature,velFeature

class featureFusionNetwork(nn.Module):
    def __init__(self, featurefusion_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(featurefusion_layer, num_layers)
        self.num_layers = num_layers
    def forward(self, src1, src2):
    # def forward(self, src1):
        output1,output2 = src1,src2
        for layer in self.layers:
            output1,output2 = layer(output1,output2)
        return output1,output2

class selfFusionNet(nn.Module):
    def __init__(self, fusion_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(fusion_layer, num_layers)

    def forward(self, src1):
        output1=src1
        for layer in self.layers:
            output1 = layer(output1)
        return output1

class segmentationModule(nn.Module):
    def reshape(self,x):
        B,HW,C=x.shape

        x = x.view(B, int(HW**(0.5)),int(HW**(0.5)),C)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    def reshape1(self,x):
        B,HW,C=x.shape
        x = x.view(B, int((HW/2)**(0.5)),int((HW/2)**(0.5)),C*2)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    def __init__(self,inChannel,feaSize):
        super().__init__()
        self.embedMagLayer1 = fusion.PatchEmbedding(feaSize,16,inChannel,inChannel)# 128 image size, 8 patch size, 64 channel
        self.embedVelLayer1 = fusion.PatchEmbedding(feaSize,16,inChannel,inChannel)

        self.feature_layer1=featureFusionLayer(inputdim=inChannel)
        self.featureNetwork1= featureFusionNetwork(self.feature_layer1,4)

        self.expandFusion=expandLayer(inputChannel=inChannel,outputChannel=inChannel)

    def forward(self,magLayer0,velLayer0):
        magLayer1Emb=self.embedMagLayer1(magLayer0)
        velLayer1Emb=self.embedVelLayer1(velLayer0)

        #featureNetwork1: cross attention for the magLayer1Emb, and velLayer1Emb
        mag,vel=self.featureNetwork1(magLayer1Emb,velLayer1Emb)

        # fusedFeature=torch.cat([mag,vel])
        # fusedFeature=self.featureFusionLayer
        # featureFusionNet: self-attention to fusion the magFeature and velFeature
        # fusedFeature=self.selfFusionNet(fusedFeature)

        mag=self.reshape(mag)
        mag=self.expandFusion(mag)
        mag=F.interpolate(mag,scale_factor=2)
        vel = self.reshape(vel)
        vel = self.expandFusion(vel)
        vel=F.interpolate(vel,scale_factor=2)
        return mag,vel

class FuFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.firstConvM=unetpart.DoubleConv(1,64)
        self.firstConvV=unetpart.DoubleConv(3,64)

        self.firstDownM=unetpart.Down(64,128)
        self.firstDownV=unetpart.Down(64,128)
        self.secondDownM=unetpart.Down(128,256)
        self.secondDownV=unetpart.Down(128,256)
        self.thirdDonwM=unetpart.Down(256,512)
        self.thirdDonwV=unetpart.Down(256,512)

        self.layer0=segmentationModule(inChannel=64,feaSize=256)
        self.layer1=segmentationModule(inChannel=128,feaSize=128)
        self.layer2=segmentationModule(inChannel=256,feaSize=64)
        self.layer3=segmentationModule(inChannel=512,feaSize=32)


    def forward(self,mag,vel):
        # m1,m2,m3,m4=self.magFeature(mag) #[1, 64, 256, 256],[1, 128, 128, 128],[1, 256, 64, 64],[1, 512, 32, 32]
        # v1,v2,v3,v4=self.velFeature(vel)
        m1=self.firstConvM(mag)
        v1=self.firstConvV(vel)
        m1New,v1New=self.layer0(m1,v1)
        m1=m1+m1New
        v1=v1+v1New

        m2=self.firstDownM(m1)
        v2=self.firstDownV(v1)
        m2New, v2New = self.layer1(m2, v2)
        m2 = m2 + m2New
        v2 = v2 + v2New

        m3 = self.secondDownM(m2)
        v3 = self.secondDownV(v2)
        m3New, v3New = self.layer2(m3, v3)
        m3 = m3 + m3New
        v3 = v3 + v3New

        m4 = self.thirdDonwM(m3)
        v4 = self.thirdDonwV(v3)
        m4New, v4New = self.layer3(m4, v4)
        m4 = m4 + m4New
        v4 = v4 + v4New


        return m1,m2,m3,m4,v1,v2,v3,v4

#part 1
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, mid_channels,out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , mid_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(mid_channels*2, out_channels)

    def forward(self, x1,x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class seg(nn.Module):
    def __init__(self):
        super().__init__()
        self.twinsTran=FuFeature()
        self.s4= nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=3,padding=1)
        self.s3= nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,padding=1)
        self.s2= nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding=1)
        self.s1= nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1)

        self.conv4 = DoubleConv(512, 512)
        self.up4_3 = Up(512, 256, 256)
        self.conv3 = DoubleConv(256, 256)
        self.up3_2 = Up(256, 128, 128)
        self.conv2 = DoubleConv(128, 128)
        self.up2_1 = Up(128, 64, 64)
        self.conv1 = DoubleConv(64, 64)
        self.upConv = DoubleConv(64, 32)
        self.outConv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self,mag,vel):
        m1,m2,m3,m4,v1,v2,v3,v4=self.twinsTran(mag,vel)
        s4=torch.cat([m4,v4],dim=1) #1024,32,32
        s4=self.s4(s4)
        s3=torch.cat([m3,v3],dim=1) #512,64,64
        s3=self.s3(s3)
        s2=torch.cat([m2,v2],dim=1) #256,128,128
        s2=self.s2(s2)
        s1=torch.cat([m1,v1],dim=1) #128,256,256
        s1=self.s1(s1)

        conv4=self.conv4(s4) #1024,32,32
        up4_3=self.up4_3(conv4,s3)# 512,64,64
        conv3=self.conv3(up4_3)# 512,16,16
        up3_2=self.up3_2(conv3,s2) # 256,32,32
        conv2=self.conv2(up3_2) #256,32,32
        up2_1=self.up2_1(conv2,s1) #64,64,64
        conv1=self.conv1(up2_1)#128,64,64
        upConv=self.upConv(conv1) # 16,128,128
        out=self.outConv(upConv)

        return torch.sigmoid(out)




