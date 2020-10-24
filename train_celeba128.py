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
parser.add_argument('--name', dest='experiment_name', default='celeba128_dim200_cd28_cc28')
args = parser.parse_args()
experiment_name = args.experiment_name+'_'+'V1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#SUPERVISED = True
SUPERVISED = False
batch_size = 16
z_dim_num = 128
c_d_num = 92
c_c_num = 36
input_size = z_dim_num+c_d_num+c_c_num 
img_channel = 3
epoch = 100
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
import network.network_celeba128 as net2

G = net2.generator_mwm().to(device)
D = net2.discriminator_mwm().to(device)

BCE_loss = nn.BCELoss()
CE_loss = nn.CrossEntropyLoss()
MSE_loss = nn.MSELoss()
G_optimizer = optim.Adam(G.parameters(),  betas=(0.5, 0.99),amsgrad=True)
D_optimizer = optim.Adam(D.parameters(), lr=0.0002,betas=(0.5, 0.99),amsgrad=True)
info_optimizer = optim.Adam(itertools.chain(G.parameters(), D.parameters()),lr=0.0001,betas=(0.6, 0.95),amsgrad=True)#G,D都更新

with open(save_root+'setting.txt', 'w') as f:
	print(z_dim_num+'-----'+c_d_num+'-----'+c_c_num+'-----',file=f)
	print('----',file=f)
	#print(G1,file=f)
	print('----',file=f)
	print(G,file=f)
	print('----',file=f)
	#print(D1,file=f)
	print('----',file=f)
	print(D,file=f)
	print('----',file=f)
	print(G_optimizer,file=f)
	print('----',file=f)
	print(D_optimizer,file=f)
	print('----',file=f)
	print(info_optimizer,file=f)
	print('----',file=f)

#----------------- Self-Supervised Constants ----------

sample_num = 64
column = 8
row = 8
gap = 0 # gap + row < c_d_num

##--------------z： 每个变量的 noise一样-----------
test_z = torch.zeros((sample_num, input_size)) #[-1,128]
sample_z = torch.rand((1, z_dim_num)).expand(sample_num, z_dim_num) # [-1,64]
test_z[:,:z_dim_num]=sample_z


##---------------z_d: 每个z_d一个one-hot位，每行一样---------
sample_z_d = torch.zeros((sample_num, c_d_num)) #[64,32]
for i in range(row):
	for j in range(column):
		sample_z_d[i*8+j,i+gap]=1
#print(z_d[8:16])
test_z[:,z_dim_num:z_dim_num+c_d_num]=sample_z_d
#print(z_a[:,64:72])

##---------------z_c: 每个z_c一个维度的连续变量，每行变化一个维度---------
sample_z_c = torch.zeros((sample_num, c_c_num)) #[64,32]
temp_c = torch.linspace(-1, 1, row)
for i in range(column):
	for j in range(row):
		sample_z_c[i*8+j,i+gap]=temp_c[j]
#print(z_c[8:16])
test_z[:,z_dim_num+c_d_num:]=sample_z_c
#print(z_a[:,96:104])

test_z = test_z.to(device)

#----------------- Training ------------
G.train()
D.train()
d_real_flag, d_fake_flag = torch.ones(batch_size).to(device), torch.zeros(batch_size).to(device)
print('training start!!')
start_time = time.time()

for i in range(epoch):
	epoch_start_time = time.time()
	j=0
	#for y, c_d_true in tqdm.tqdm(data_loader):
	for y in tqdm.tqdm(data_loader):
	#for j in range(5000):
		j = j + 1
		z = torch.rand((batch_size, z_dim_num)).to(device)
		if SUPERVISED == True:
			c_d = torch.zeros((batch_size, c_d_num)).scatter_(1, c_d_true.type(torch.LongTensor).unsqueeze(1), 1)
		else:
			c_d = torch.from_numpy(np.random.multinomial(1, c_d_num * [float(1.0 / c_d_num)],size=[batch_size])).type(torch.FloatTensor)#投骰子函数,随机化y_disc_
		c_d = c_d.to(device)
		c_c = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size, c_c_num))).type(torch.FloatTensor).to(device)
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
		#latend_c = torch.cat([z, c_c, c_d], 1).to(device)
		y = y.to(device)
		y_f = G(z, c_c, c_d)
		#print(y.shape)
		#print(y_f.shape)
		d_t = D(y)
		D_real = torch.sigmoid(d_t[:, z_dim_num])
		d_f = D(y_f)
		D_fake = torch.sigmoid(d_f[:, z_dim_num])
		D_real_loss = BCE_loss(D_real, d_real_flag)#1
		D_fake_loss = BCE_loss(D_fake, d_fake_flag)#0
		D_loss = D_real_loss + D_fake_loss
		train_hist['D_loss'].append(D_loss.item())
		D_loss.backward(retain_graph=True)
		D_optimizer.step()

# update G network
		G_optimizer.zero_grad()
		d_f_2 = D(y_f)
		D_fake_2 = torch.sigmoid(d_f_2[:, z_dim_num])
		G_loss = BCE_loss(D_fake_2, d_real_flag)
		train_hist['G_loss'].append(G_loss.item())
		G_loss.backward(retain_graph=True)
		G_optimizer.step()

# information loss
		D_optimizer.zero_grad() #这两个网络不清零，梯度就会乱掉,训练失败
		G_optimizer.zero_grad()
		y_info = G(latend_c)
		d_f_3 = D(y_info)
		D_disc_info, D_cont_info = d_f_3[:,z_dim_num:z_dim_num + c_d_num], d_f_3[:, z_dim_num + c_d_num:]
		disc_loss = CE_loss(D_disc_info, torch.max(c_d, 1)[1])#第二个是将Label由one-hot转化为10进制数组
		cont_loss = (D_cont_info - c_c)**2
		info_loss = disc_loss + cont_loss*c_c_num
		info_loss = info_loss.mean()
		train_hist['info_loss'].append(info_loss.item())
		info_loss.backward()
		info_optimizer.step()
# test
		train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
		if ((j + 1) % 100) == 0:
			with open(save_root+'setting.txt', 'a') as f:
				print('----',file=f)
				print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, info_loss: %.8f" %((i + 1), (j + 1), data_loader.dataset.__len__() // batch_size, D_loss.item(), G_loss.item(), info_loss.item()),file=f)
				print('----',file=f)
			with torch.no_grad():# save2img
				G.eval()
				D.eval()
				image_frame_dim = int(np.floor(np.sqrt(sample_num)))
				samples = G(test_z)
				samples = (samples + 1) / 2
				torchvision.utils.save_image(samples, save_dir+'/%d_%d_Epoch—d_c.png' % (i,j), nrow=8)
				test_z2 = D(samples)
				samples2 = G(test_z2)
				img = torch.cat((samples[:8],samples2[:8]))
				img = (img + 1) / 2
				torchvision.utils.save_image(img, save_dir + '/%d_%d_Epoch-rc.png' % (i,j), nrow=8)
				train_hist['total_time'].append(time.time() - start_time)
				with open(save_root+'lossAll.txt', 'a') as f:
						print('----',file=f)
						print(train_hist,file=f)
						print('----',file=f)
	print('epoch:'+str(i))
	if i%3 == 0:
		torch.save({'epoch': epoch + 1,'G': G.state_dict()},'%s/Epoch_(%d).pth' % (ckpt_dir, epoch + 1))#save model
		torch.save({'epoch': epoch + 1,'D': D.state_dict()},'%s/Epoch_(%d).pth' % (ckpt_dir, epoch + 1))#save model

#loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)