import torch
import torch.nn as nn
import torch.nn.functional as F
      
class TransferConv_n(nn.Module):
    def __init__(self, resnet, in_c):
        super().__init__()
        if resnet:
            in_c = in_c // 2
            self.layer1 = nn.Sequential(
                        nn.Conv2d(in_c // 2, in_c // 2, kernel_size=1, padding=0),  # 使用 1x1 卷积降低通道
                        nn.BatchNorm2d(in_c // 2, momentum=0.1, affine=True),
                        nn.LeakyReLU(0.2, True),
                    )
                    
            self.layer2 = nn.Sequential(
                        nn.Conv2d(in_c // 2, in_c, kernel_size=1, padding=0),  # 使用 1x1 卷积降低通道
                        nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d(2)
                    )#200 320 10 10            
            self.layer3 = nn.Sequential(
                        nn.Conv2d(in_c , in_c*2, kernel_size=1, padding=0),  # 使用 3x3 卷积增加到 640
                        nn.BatchNorm2d(in_c * 2, momentum=0.1, affine=True),
                        nn.LeakyReLU(0.2, True),
                    )
            self.layer4 = nn.Sequential(
                nn.Conv2d(in_c * 2, in_c *2, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_c *2, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
            )

        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_c, in_c // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c // 2, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(in_c, in_c // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c // 2, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
            )
   
    def forward(self, x):
        output = self.layer1(x) 
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return output     
      
class TransferConv_m(nn.Module):
    def __init__(self, resnet, in_c):
        super().__init__()
        if resnet:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c // 2, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_c // 2, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
            )
      
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_c, in_c // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c // 2, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
            )
    
    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        return output

class TransferConv_h(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),

        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1,padding=0),
            nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        return output
        

class MFI(nn.Module):
    def __init__(self, resnet, in_c):
        super().__init__()
        self.transferconv_h = TransferConv_h(in_c)
        self.transferconv_m = TransferConv_m(resnet, in_c)
        self.transferconv_n = TransferConv_n(resnet, in_c)

        # self.alpha = nn.Parameter(torch.tensor(0.5))

        # self.alpha_nm = nn.Parameter(torch.tensor(0.3))  # 初始值
        # self.alpha_mh = nn.Parameter(torch.tensor(0.3))
        # self.alpha_h = nn.Parameter(torch.tensor(0.3))

    def reconstructing_procedure(self, f_h, f_m,f_n):
        _, c, h, w = f_h.shape
        f_h = f_h.view(f_h.size(0), f_h.size(1), -1)#600 64 25
        f_m = f_m.view(f_m.size(0), f_m.size(1), -1)#600 64 25
        f_n = f_h.view(f_n.size(0), f_n.size(1), -1)#600 64 25
        #3.f_n重构f_m 2->3
        f_n_T = torch.transpose(f_n, 2, 1) #f_n转置
        matrix_nm = torch.matmul(f_n_T, f_m) #相似矩阵
        l2_n = torch.norm(matrix_nm)#归一化
        matrix_nm = torch.tanh(matrix_nm / l2_n)
        f_refine_nm = torch.matmul(f_n, matrix_nm) + f_m

        ##f_m重构f_h 3->4 
        f_m_T = torch.transpose(f_refine_nm, 2, 1) #600 25 64   (fh600 64 25 * fm_t转置(600,25,64)= hm600,25,25)
        matrix_hm = torch.matmul(f_m_T, f_h) #600 25 25 #相似矩阵
        l2_m = torch.norm(matrix_hm) #矩阵 归一化
        matrix_hm = torch.tanh(matrix_hm / l2_m) #重建特征 f_refine_h
        f_refine_mh = torch.matmul(f_refine_nm, matrix_hm) + f_h
        
        #f_refine_mh 4 自重构
        f_h_T = torch.transpose(f_refine_mh, 2, 1)
        matrix_hh = torch.matmul(f_h_T, f_h)
        l2_h = torch.norm(matrix_hh)
        matrix_hh = torch.tanh(matrix_hh / l2_h)
        f_refine_h = torch.matmul(f_refine_mh, matrix_hh) + f_h
 
        f_refine_nm = f_refine_nm.view(-1, c, h, w)
        f_refine_mh = f_refine_mh.view(-1, c, h, w)
        f_refine_h = f_refine_h.view(-1, c, h, w)
        

        return f_refine_nm,f_refine_mh, f_refine_h
    ##fh 400,64,5,5 fm 400 64 10 10 fn [400, 64, 21, 21]
    def forward(self, f_h, f_m, f_n):
        f_h = self.transferconv_h(f_h) #400 64 5 5  200 640 5 5
        f_m = self.transferconv_m(f_m) #400 64 5 5  200 640 5 5
        f_n = self.transferconv_n(f_n) #400 64 5 5  200 640 5 5
        # l2_diff = torch.norm(f_m - f_h, p=2)
        # print(f"L2 norm difference between f_m and f_h: {l2_diff.item()}")
        f_refine_nm,f_refine_mh, f_refine_h = self.reconstructing_procedure(f_h, f_m, f_n)
        

        return f_refine_nm,f_refine_mh, f_refine_h
