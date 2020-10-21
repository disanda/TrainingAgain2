import torch
import torch.optim as optim
import torchvision
import torch
import torch.nn as nn
import os
import numpy as np
import itertools
import argparse
from PIL import Image
import time
import tqdm
import random
import data

#--------------- Setting Params ---------

parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='experiment_name', default='celeba256_dim256_AG_CE')
args = parser.parse_args()
experiment_name = args.experiment_name+'_'+'V1'

gpu_mode = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#SUPERVISED = True
SUPERVISED = False
batch_size = 32
z_dim_num = 128
c_d_num = 92
c_c_num = 36
input_size = z_dim_num+c_d_num+c_c_num 
img_channel = 3
epoch = 10
img_size = 256 

if not os.path.exists('./info_output/'):
    os.mkdir('./info_output/')

save_root='./info_output/%s/'%experiment_name
if not os.path.exists('./info_output/%s/'% experiment_name):
    os.mkdir('./info_output/%s/'% experiment_name)


save_dir = './info_output/%s/sample_training/' % experiment_name
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

ckpt_dir = './info_output/%s/checkpoints/' % experiment_name
if not os.path.exists(ckpt_dir):
	os.mkdir(ckpt_dir)

train_hist = {}
train_hist['D_loss'] = []
train_hist['G_loss'] = []
train_hist['info_loss'] = []
train_hist['per_epoch_time'] = []
train_hist['total_time'] = []

#dataSets
data_loader, shape = data.make_dataset(dataset_name='celebaHQ', batch_size=batch_size, img_size=img_size, pin_memory=True)


#---------------- Pre-Model ------------
#-----DCGAN celebaA---------## input_dim=256, Gscale=8, Dscale=4

import network.network_1 as net1
import network.network_1_SSencoder as net2

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

netG1 = net1.Generator(input_dim=256, output_channels = 3, image_size=256, scale=8)# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
pathG1 = 'F:/pre-model/spectral_norm_celeba256_inputDim256_Scale8_4/checkpoints/Epoch_G_(99).pth'
netG1.load_state_dict(torch.load(pathG1,map_location=device)) #shadow的效果要好一些 
netD1 = net1.Discriminator_SpectrualNorm(input_dim=256, input_channels = 3, image_size=256, scale=4)# in: [-1,3,1024,1024],out:[], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
pathD1 = 'F:/pre-model/spectral_norm_celeba256_inputDim256_Scale8_4/checkpoints/Epoch_D_(99).pth'
netD1.load_state_dict(torch.load(pathD1,map_location=device))

netG2 = net2.Generator_SS()
netD2 = net2.Discriminator_SS()

#----------- param load ------------
toggle_grad(netG1,False)
toggle_grad(netG2,False)
paraDictG1 = dict(netG1.named_parameters()) # pre_model weight dict
for i,j in netG2.named_parameters():
	if i in paraDictG1.keys():
		w = paraDict[i]
		j.copy_(w)
toggle_grad(netG2,True)

toggle_grad(netD1,False)
toggle_grad(netD2,False)
paraDictD1 = dict(netD1.named_parameters()) # pre_model weight dict
for i,j in netD2.named_parameters():
	if i in paraDictD1.keys():
		w = paraDict[i]
		j.copy_(w)
toggle_grad(netD2,True)

G_optimizer = optim.Adam(G2.parameters(),  betas=(0.5, 0.99),amsgrad=True)
D_optimizer = optim.Adam(D2.parameters(), lr=0.0002,betas=(0.5, 0.99),amsgrad=True)
info_optimizer = optim.Adam(itertools.chain(G2.parameters(), D2.parameters()),lr=0.0001,betas=(0.6, 0.95),amsgrad=True)#G,D都更新

with open(save_root+'setting.txt', 'w') as f:
	print('----',file=f)
	print(G1,file=f)
	print('----',file=f)
	print(G2,file=f)
	print('----',file=f)
	print(D1,file=f)
	print('----',file=f)
	print(D2,file=f)
	print('----',file=f)
	print(G_optimizer,file=f)
	print('----',file=f)
	print(D_optimizer,file=f)
	print('----',file=f)
	print(info_optimizer,file=f)
	print('----',file=f)

#----------------- Self-Supervised Constants ----------
#这是个排列组合的问题
c_d_num_t = 8
c_c_num_t = 2
c_c_scale_t = 10 # 一个连续变量c_c中的刻度
sample_num =c_d_num*c_c_num*c_c_scale

sample_z = torch.rand((1, z_dim_num)).expand(sample_num, z_dim_num) #每个样本的noize相同

sample_d = torch.zeros(sample_num, c_d_num_t)#[-1,c_d]

sample_c = torch.zeros(sample_num, c_c_num_t)#[-1,c_c]
temp_c = torch.linspace(-1, 1, c_c_scale_t)		#-1->1的等差数列

for i in range(c_d_num_t):
	sample_d[ i*c_c_num_t*c_c_scale_t: (i+1)*c_c_num_t*c_c_scale_t, i]=1
	for j in range(c_c_num_t):
		x = i*c_c_num_t
		sample_c[ (x+j)*c_c_scale_t: (x+j+1)*c_c_scale_t , j ]=temp_c

test_z = torch.cat([sample_z, sample_d, sample_c], 1).to(device)

#sample_c[ 3*c_c_scale: (4)*c_c_scale , 0 ]=temp_c
# print('------------')
# print(sample_z.shape) # -1 32
# print(sample_d.shape) # -1 2
# print(sample_c.shape)
# print(sample_z)
# print(sample_d)
# print(sample_c)
# #print(sample_z2[1])
# z = torch.cat([sample_z, sample_d, sample_c], 1)
# for i in range(5):
# 	print(z[i])


#----------------- Training ------------
G.train()
D.train()
d_real_flag, d_fake_flag = torch.ones(batch_size), torch.zeros(batch_size)
print('training start!!')
start_time = time.time()

for i in range(epoch):
	epoch_start_time = time.time()
	j=0
	#for y, c_d_true in tqdm.tqdm(data_loader):
	for y in tqdm.tqdm(data_loader):
	#for j in range(5000):
		j = j + 1
		z = torch.rand((batch_size, z_dim_num))
		if SUPERVISED == True:
			c_d = torch.zeros((batch_size, c_d_num)).scatter_(1, c_d_true.type(torch.LongTensor).unsqueeze(1), 1)
		else:
			c_d = torch.from_numpy(np.random.multinomial(1, c_d_num * [float(1.0 / c_d_num)],size=[batch_size])).type(torch.FloatTensor)#投骰子函数,随机化y_disc_
		c_c = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size, c_c_num))).type(torch.FloatTensor)
# update D1 network
		# D_optimizer.zero_grad()
		# y_f = G(z, c_c, c_d)
		# D_real, _, _ = D(y)
		# D_fake, _, _ = D(y_f)
		# D_real_loss = BCE_loss(D_real, d_real_flag)#1
		# D_fake_loss = BCE_loss(D_fake, d_fake_flag)#0
		# #D_real_loss, D_fake_loss = d_loss_fn(D_real, D_fake)
		# #gp = loss_norm_gp.gradient_penalty(D, y, y_f, mode=gp_mode)
		# #gp = loss_norm_gp.gradient_penalty(functools.partial(D), y, y_f, gp_mode='0-gp', sample_mode='line')
		# gp=0
		# D_loss = D_real_loss + D_fake_loss + gp
		# train_hist['D_loss'].append(D_loss.item())
		# D_loss.backward(retain_graph=True)
		# D_optimizer.step()

# updata D2 network v2: 不用GT
		# D_optimizer.zero_grad()
		# latend_c = torch.cat([z, c_c, c_d], 1).to(device)
		# y_f = G2(latend_c)
		# D_fake, D_fake_d, D_fake_c = D(y_f)
		# y_f = G2(latend_c)
		# D_real_loss = BCE_loss(D_real, d_real_flag)#1
		# D_fake_loss = BCE_loss(D_fake, d_fake_flag)#0
		# D_loss = D_real_loss + D_fake_loss
		# train_hist['D_loss'].append(D_loss.item())
		# D_loss.backward(retain_graph=True)
		# D_optimizer.step()

# update D2 network
		D_optimizer.zero_grad()
		latend_c = torch.cat([z, c_c, c_d], 1).to(device)
		y = y.to(device)
		y_f = G(latend_c)
		D_real, _, _ = D()
		D_fake, _, _ = D(y_f)
		D_real_loss = BCE_loss(D_real, d_real_flag)#1
		D_fake_loss = BCE_loss(D_fake, d_fake_flag)#0
		D_loss = D_real_loss + D_fake_loss + gp
		train_hist['D_loss'].append(D_loss.item())
		D_loss.backward(retain_graph=True)
		D_optimizer.step()

# update G network
		G_optimizer.zero_grad()
		y_f = G(z, c_c, c_d)
		D_fake,D_disc,D_cont = D(y_f)
		G_loss = BCE_loss(D_fake, d_real_flag)
		#G_loss = g_loss_fn(D_fake)
		train_hist['G_loss'].append(G_loss.item())
		G_loss.backward(retain_graph=True)
		G_optimizer.step()
# information loss
		D_optimizer.zero_grad() #这两个网络不清零，梯度就会乱掉,训练失败
		G_optimizer.zero_grad()
		y_info = G(z, c_c, c_d)
		_,D_disc_info,D_cont_info = D(y_info)
		disc_loss = CE_loss(D_disc_info, torch.max(c_d, 1)[1])#第二个是将Label由one-hot转化为10进制数组
		cont_loss = (D_cont_info - c_c)**2
		info_loss = disc_loss + cont_loss*c_c_num
		info_loss = info_loss.mean()
		train_hist['info_loss'].append(info_loss.item())
		info_loss.backward()
		info_optimizer.step()
		train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
		if ((j + 1) % 100) == 0:
			with open(save_root+'setting.txt', 'a') as f:
				print('----',file=f)
				print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, info_loss: %.8f" %((i + 1), (j + 1), train_loader.dataset.__len__() // batch_size, D_loss.item(), G_loss.item(), info_loss.item()),file=f)
				print('----',file=f)
	print('epoch:'+str(i))
# save2img
	with torch.no_grad():
		G1.eval()
		D2.eval()
		image_frame_dim = int(np.floor(np.sqrt(sample_num)))
		samples = G2(test_z)
		samples = (samples + 1) / 2
		torchvision.utils.save_image(samples, save_dir+'/%d_Epoch—d_c.png' % i, nrow=10)
		a,b,c = D2(samples)
		test_z2 = torch.cat([a, b, c], 1)
		samples2 = G2(test_z2)
		img = torch.cat((samples[:8],samples2[:8]))
		img = (img + 1) / 2
		torchvision.utils.save_image(img, save_dir + '/%d_Epoch-rc.png' % i, nrow=8)
		train_hist['total_time'].append(time.time() - start_time)
		with open(save_root+'lossAll.txt', 'a') as f:
				print('----',file=f)
				print(train_hist,file=f)
				print('----',file=f)
	if i%3 == 0:
		torch.save({'epoch': epoch + 1,'G': G2.state_dict()},'%s/Epoch_(%d).pth' % (ckpt_dir, epoch + 1))#save model
		torch.save({'epoch': epoch + 1,'D': D2.state_dict()},'%s/Epoch_(%d).pth' % (ckpt_dir, epoch + 1))#save model

#loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)