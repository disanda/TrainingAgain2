import torch

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

#-----DCGAN celebaA---------
## input_dim=256, Gscale=8, Dscale=4

import network.network_1 as net1
import network.network_1_SSencoder as net2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# netG1 = net1.Generator(input_dim=256, output_channels = 3, image_size=256, scale=8)# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
# pathG1 = '/home/disanda/Desktop/celebaGAN/output/spectral_norm_celeba256_inputDim256_Scale8_4/checkpoints/Epoch_G_(99).pth'
# netG1.load_state_dict(torch.load(pathG1,map_location=device)) #shadow的效果要好一些 
# netD1 = net1.Discriminator_SpectrualNorm(input_dim=256, input_channels = 3, image_size=256, scale=4)# in: [-1,3,1024,1024],out:[], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
# pathD1 = '/home/disanda/Desktop/celebaGAN/output/spectral_norm_celeba256_inputDim256_Scale8_4/checkpoints/Epoch_D_(99).pth'
# netD1.load_state_dict(torch.load(pathD1,map_location=device))


#netG2 = net2.Generator_SS()
# netD2 = net2.Discriminator_SS()

# print(netG1)
# print(dict(netG1.named_parameters()).keys())
# print(netG2)
# print(dict(netG2.named_parameters()).keys())

#----------- param load ------------
# toggle_grad(netG1,False)
# toggle_grad(netG2,False)
# paraDictG1 = dict(netG1.named_parameters()) # pre_model weight dict
# for i,j in netG2.named_parameters():
# 	if i in paraDictG1.keys():
# 		w = paraDictG1[i]
# 		j.copy_(w)
# toggle_grad(netG2,True)

# toggle_grad(netD1,False)
# toggle_grad(netD2,False)
# paraDictD1 = dict(netD1.named_parameters()) # pre_model weight dict
# for i,j in netD2.named_parameters():
# 	if i in paraDictD1.keys():
# 		w = paraDict[i]
# 		j.copy_(w)
# toggle_grad(netD2,True)


#---------test input & output---------
# z = torch.randn(5,256)
# x = netG2(z)
# print(x.shape)
# z_ = netD2(x)
# print(dict(netG2.named_parameters()).keys())

#---------去除指定层---------
#这里需要去除net的前三层
netG1 = net1.Generator(input_dim=256, output_channels = 3, image_size=256, scale=8)# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
pathG1 = '/Users/apple/Desktop/pre_trained_pytorch/spectral_norm_celeba256_inDim256_Scale_8_4/G_EP_99.pth'
netG1.load_state_dict(torch.load(pathG1,map_location=device)) 

netG2 = net2.Generator_SS()

# import copy
# dict1 = copy.copy(G1_state_dict)
# print(dict1.keys())
dict1 = dict(netG1.named_parameters())
dict2 = dict(netG2.named_parameters())

print(type(dict1))
print(type(dict2))

keys = []
dict3 = {}

for i,j in dict1.items():
	if i.startswith('fc'):
		print(i)
		continue
	keys.append(i)

dict3 = {k:dict1[k] for k in keys}
print(dict3.keys())

# G2.state_dict().updata(dict2)

# torch.save(net.state_dict(),'')


#---------torch 1.6 转低版本 ,即关闭zip序列化-----
# netG1 = net1.Generator(input_dim=256, output_channels = 3, image_size=256, scale=8)# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
# pathG1 = '/home/disanda/Desktop/celebaGAN/output/spectral_norm_celeba256_inputDim256_Scale8_4/checkpoints/Epoch_G_(99).pth'
# netG1.load_state_dict(torch.load(pathG1,map_location=device)) #shadow的效果要好一些 
# torch.save(netG1.state_dict(),'/home/disanda/Desktop/celebaGAN/output/spectral_norm_celeba256_inputDim256_Scale8_4/G_EP_99.pth',_use_new_zipfile_serialization=False)

# netD1 = net1.Discriminator_SpectrualNorm(input_dim=256, input_channels = 3, image_size=256, scale=4)# in: [-1,3,1024,1024],out:[], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
# pathD1 = '/home/disanda/Desktop/celebaGAN/output/spectral_norm_celeba256_inputDim256_Scale8_4/checkpoints/Epoch_D_(99).pth'
# netD1.load_state_dict(torch.load(pathD1,map_location=device))
# torch.save(netD1.state_dict(),'/home/disanda/Desktop/celebaGAN/output/spectral_norm_celeba256_inputDim256_Scale8_4/D_EP_99.pth',_use_new_zipfile_serialization=False)


