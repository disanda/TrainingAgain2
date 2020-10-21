import torch

z_dim_num= 64
c_d_num = 32
c_c_num = 32
dim_all = 128

sample_num = 64
column = 8
row = 8
gap = 0 # gap + row < c_d_num

#--------------z： 每个变量的 noise一样-----------
z_a = torch.zeros((sample_num, dim_all)) #[-1,128]
z = torch.rand((1, z_dim_num)).expand(sample_num, z_dim_num) # [-1,64]
z_a[:,:z_dim_num]=z
# print(z_a)

#---------------z_d: 每个z_d一个one-hot位，每行一样---------
z_d = torch.zeros((sample_num, c_d_num)) #[64,32]
for i in range(row):
	for j in range(column):
		z_d[i*8+j,i+gap]=1
#print(z_d[8:16])
z_a[:,z_dim_num:z_dim_num+c_d_num]=z_d
#print(z_a[:,64:72])

#---------------z_c: 每个z_c一个维度的连续变量，每行变化一个维度---------
z_c = torch.zeros((sample_num, c_d_num)) #[64,32]
temp_c = torch.linspace(-1, 1, row)
for i in range(column):
	for j in range(row):
		z_c[i*8+j,i+gap]=temp_c[j]
#print(z_c[8:16])
z_a[:,z_dim_num+c_d_num:]=z_c
#print(z_a[:,96:104])

for i in range(8,16):
	print(z_a[i,64:])



# 固定noise和cc，每c_d个变一次c_d
# sample_z = torch.zeros((sample_num, z_dim_num))
# temp = torch.zeros((c_d_num, 1))
# for i in range(sample_num//c_d_num):
# 	sample_z[i * c_d_num] = torch.rand(1, z_dim_num)#为连续c_d个的样本赋值。
# 	for j in range(c_d_num):
# 		sample_z[i * c_d_num + j] = sample_z[i * c_d_num]#相同c_d的noize都相同

# for i in range(c_d_num):
# 	temp[i, 0] = i #每一个标签
# temp_d = torch.zeros((sample_num, 1))
# for i in range(sample_num//c_d_num):
# 	temp_d[i * c_d_num: (i + 1) * c_d_num] = temp[i%c_d_num] #每c_d个的d一样
# sample_d = torch.zeros((sample_num, c_d_num)).scatter_(1, temp_d.type(torch.LongTensor), 1)
# sample_c = torch.zeros((sample_num, c_c_num))

# i = 6
# print(sample_z[i])
# print(sample_d[i])
# print(sample_c[i])
# print(sample_z.shape)
# print(sample_d.shape)
# print(sample_c.shape)

# 观察单一变量，固定其他变量
# sample_z2 = torch.rand((1, z_dim_num)).expand(sample_num, z_dim_num) #每个样本的noize相同
# sample_d2 = torch.zeros(sample_num, c_d_num)#[200,c_d]

# temp_c = torch.linspace(-1, 1, c_d_num)		#c_d_num个范围在-1->1的等差数列
# sample_c2 = torch.zeros((sample_num, c_c_num))#[200,c_c]

# for i in range(sample_num//c_d_num):		#每c_d个noise,c_d相同,c_c不同
# 	#d_label = random.randint(0,c_d_num-1)
# 	d_label = i%c_d_num
# 	sample_d2[i*c_d_num:(i+1)*c_d_num, d_label] = 1
# 	sample_c2[i*c_d_num:(i+1)*c_d_num,i%c_c_num] = temp_c

# i = 6
# print(sample_z2[i])
# print(sample_d2[i])
# print(sample_c2[i])
# print(sample_z2.shape)
# print(sample_d2.shape)
# print(sample_c2.shape)
