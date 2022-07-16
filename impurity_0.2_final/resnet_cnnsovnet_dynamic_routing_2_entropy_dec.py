import torch
from torch import nn
from torch.nn import functional as F
from utils import *
from constants import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # stride 1: 64 -> 64 | stride 2: 64 -> 32
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # 64 -> 64 ie same
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.selu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.selu(out)
        return out

class ResNetPreCapsule(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNetPreCapsule, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)#(b_size,16,32,32)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)#(b_size,16,32,32)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)#(b_size,32,16,16)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)#(b_size,64,8,8)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.selu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #print(f"out.shape : {out.shape}")
        #assert False
        return out

def convertToCaps(x):
    return x.unsqueeze(2)

class PrimaryCapsules(nn.Module):
    def __init__(self,in_channels,num_capsules,out_dim,H=16,W=16):
        super(PrimaryCapsules,self).__init__()
        self.in_channels = in_channels
        self.num_capsules = num_capsules
        self.out_dim = out_dim
        self.H = H
        self.W = W
        self.preds = nn.Sequential(nn.Conv2d(in_channels,num_capsules*out_dim,kernel_size=1),
                                   nn.SELU(),
                                   #nn.BatchNorm2d(num_capsules*out_dim))
                                   nn.LayerNorm((num_capsules*out_dim,H,W))
                                  )

    def forward(self,x):
        # x : (b,64,16,16)
        primary_capsules = self.preds(x) #(b,16*8,16,16)
        primary_capsules = primary_capsules.view(-1,self.num_capsules,self.out_dim,self.H,self.W)
        return primary_capsules #(b,16,8,16,16)

class ConvCapsule(nn.Module):
    def __init__(self,in_caps,in_dim,out_caps,out_dim,kernel_size,stride,padding):
        super(ConvCapsule,self).__init__()
        self.in_caps = in_caps
        self.in_dim = in_dim
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.preds = nn.Sequential(nn.Conv2d(in_dim,out_caps*out_dim,kernel_size=kernel_size,stride=stride,padding=padding),
                                   nn.BatchNorm2d(out_caps*out_dim),
                                   nn.SELU())
     
    def forward(self,in_capsules,ITER=3):
        # in_capsules : (b,16,8,16,16)
        batch_size, _, _, H, W = in_capsules.size()
        in_capsules = in_capsules.view(batch_size*self.in_caps,self.in_dim,H,W) #(b*16,8,16,16)
        predictions = self.preds(in_capsules) # (b,)
        _,_, H, W = predictions.size()
        predictions = predictions.view(batch_size, self.in_caps, self.out_caps*self.out_dim, H, W)
        predictions = predictions.view(batch_size, self.in_caps, self.out_caps, self.out_dim, H, W)
        out_capsules, cij = self.dynamic_routing(predictions,ITER)
        return out_capsules, cij

    def squash(self, inputs, dim):
        norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
        scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
        return scale * inputs

    def dynamic_routing(self,predictions,ITER=3):
        batch_size,_,_,_, H, W = predictions.size()
        b_ij = torch.zeros(batch_size,self.in_caps,self.out_caps,1,H,W).to(DEVICE)
        for it in range(ITER):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * predictions).sum(dim=1, keepdim=True)
            v_j = self.squash(inputs=s_j, dim=3)
            if it < ITER - 1: 
               delta = (predictions * v_j).sum(dim=3, keepdim=True)
               b_ij = b_ij + delta
        return v_j.squeeze(dim=1), c_ij


class Mask_CID(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, x, target=None):
        # x.shape = (batch, classes, dim)
        # one-hot required
        #print(f"x.shape : {x.shape} | target.shape : {target.shape}")
        if target is None:
            classes = torch.norm(x, dim=2)
            max_len_indices = classes.max(dim=1)[1].squeeze()
        else:
            max_len_indices = target.max(dim=1)[1]
        
        #print("max_len_indices: ", max_len_indices)
        increasing = torch.arange(start=0, end=x.shape[0]).cuda()
        m = torch.stack([increasing, max_len_indices], dim=1)
        
        masked = torch.zeros((x.shape[0], 1) + x.shape[2:])
        #print(f"masked.shape : {masked.shape}")
        for i in increasing:
            masked[i] = x[m[i][0], m[i][1], :].unsqueeze(0)

        return masked.squeeze(-1), max_len_indices  # dim: (batch, 1, capsule_dim)


class Decoder_smallNorb(nn.Module):
    def __init__(self, caps_size=16, num_caps=1, img_size=32, img_channels=1):
        super().__init__()
        
        self.num_caps = num_caps
        self.img_channels = img_channels
        self.img_size = img_size

        self.dense = torch.nn.Linear(caps_size*num_caps, 16*24*24).cuda(DEVICE)
        self.relu = nn.ReLU(inplace=True)
                                     
        
        self.reconst_layers1 = nn.Sequential(nn.BatchNorm2d(num_features=16, momentum=0.8),
                                            
                                            nn.ConvTranspose2d(in_channels=16, out_channels=64, 
                                                               kernel_size=3, stride=1, padding=1
                                                              )
                                            )
        
        self.reconst_layers2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, 
                                                  kernel_size=3, stride=2, padding=1
                                                 )
                               
        self.reconst_layers3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, 
                                                  kernel_size=3, stride=2, padding=1
                                                 )
                                            
        self.reconst_layers4 = nn.ConvTranspose2d(in_channels=16, out_channels=1, 
                                                  kernel_size=3, stride=1, padding=1
                                                 )
                                            
        
        self.reconst_layers5 = nn.ReLU()
                                               
    def forward(self, x):
        # x.shape = (batch, 1, capsule_dim(=32 for MNIST))
        batch = x.shape[0]
        x = x.type(torch.FloatTensor)
        x = x.to(DEVICE)
        x = self.dense(x)
        x = self.relu(x)
        x = x.reshape(-1, 16, 24, 24)
        x = self.reconst_layers1(x)        
        x = self.reconst_layers2(x)
        # padding
        p2d = (1, 0, 1, 0)
        x = func.pad(x, p2d, "constant", 0)
        x = self.reconst_layers3(x)

        # padding
        p2d = (1, 0, 1, 0)
        x = func.pad(x, p2d, "constant", 0)
        x = self.reconst_layers4(x)      
        
        x = self.reconst_layers5(x)
        x = x.reshape(-1, 1, self.img_size, self.img_size)
        
        return x  # dim: (batch, 1, imsize, imsize)


class ResnetCnnsovnetDynamicRouting(nn.Module):
    def __init__(self):
        super(ResnetCnnsovnetDynamicRouting,self).__init__()
        self.resnet_precaps = ResNetPreCapsule(BasicBlock,[2,2,2])
        self.primary_caps = PrimaryCapsules(64,16,8,24,24)#for cifar10, H,W = 16, 16. For MNIST etc. H,W = 14,14.
        self.conv_caps1 = ConvCapsule(in_caps=16,in_dim=8,out_caps=32,out_dim=16,kernel_size=3,stride=2,padding=1) # (12,12)
        self.conv_caps2 = ConvCapsule(in_caps=32,in_dim=16,out_caps=16,out_dim=16,kernel_size=3,stride=2,padding=1) # (6,6)
        self.conv_caps3 = ConvCapsule(in_caps=16,in_dim=16,out_caps=16,out_dim=16,kernel_size=3,stride=2,padding=1) # (3,3)
        self.class_caps = ConvCapsule(in_caps=16,in_dim=16,out_caps=5,out_dim=16,kernel_size=3,stride=2,padding=0) # (1,1)
        #self.conv_caps3 = ConvCapsule(in_caps=32,in_dim=16,out_caps=32,out_dim=16,kernel_size=3,stride=1,padding=0) # (3,3)
        #self.class_caps = ConvCapsule(in_caps=32,in_dim=16,out_caps=5,out_dim=16,kernel_size=3,stride=1,padding=0) # (1,1)
        self.linear = nn.Linear(16,1)
        self.mask = Mask_CID()
        self.dec = Decoder_smallNorb(caps_size=16, num_caps=1, img_size=96, img_channels=1)


    def forward(self,x,target=None):
        #print(f"\n\nx.shape : {x.shape}")
        resnet_output = self.resnet_precaps(x)
        #print(f"resnet_output.shape : {resnet_output.shape}")
        primary_caps = self.primary_caps(resnet_output)
        #print(f"primary_caps.shape : {primary_caps.shape}")
        
        conv_caps1, cij1 = self.conv_caps1(primary_caps)
        #print(f"conv_caps1.shape : {conv_caps1.shape}")
        
        conv_caps2, cij2 = self.conv_caps2(conv_caps1)
        #h = conv_caps2.register_hook(self.act_hook)
        #print(f"conv_caps2.shape : {conv_caps2.shape}")
        
        conv_caps3, cij3 = self.conv_caps3(conv_caps2)
        #print(f"conv_caps3.shape : {conv_caps3.shape}")
        
        class_caps, cij4 = self.class_caps(conv_caps3)
        #print(f"class_caps.shape : {class_caps.shape}")
        
        class_caps = class_caps.squeeze()
        #print(f"class_caps.shape : {class_caps.shape}")
        class_predictions = self.linear(class_caps).squeeze()
        #print(f"class_predictions.shape : {class_predictions.shape}")
        #assert False
        #masked, _ = self.mask(class_caps,target)
        #rec = self.dec(masked)
        return class_predictions, cij1, cij2, cij3, cij4
