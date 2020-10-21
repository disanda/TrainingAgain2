import torch

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

#-----DCGAN celebaA---------
## input_dim=256, Gscale=8, Dscale=4

import network.network_1 as net1
import network.network_1_SSencoder as net2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netG1 = net1.Generator(input_dim=256, output_channels = 3, image_size=256, scale=8)# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
pathG1 = '/home/disanda/Desktop/celebaGAN/output/spectral_norm_celeba256_inputDim256_Scale8_4/checkpoints/Epoch_G_(99).pth'
netG1.load_state_dict(torch.load(pathG1,map_location=device)) #shadow的效果要好一些 
netD1 = net1.Discriminator_SpectrualNorm(input_dim=256, input_channels = 3, image_size=256, scale=4)# in: [-1,3,1024,1024],out:[], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
pathD1 = '/home/disanda/Desktop/celebaGAN/output/spectral_norm_celeba256_inputDim256_Scale8_4/checkpoints/Epoch_D_(99).pth'
netD1.load_state_dict(torch.load(pathD1,map_location=device))


netG2 = net2.Generator_SS()
netD2 = net2.Discriminator_SS()

print(G1)
print(G2)

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
# print(z_.shape)
