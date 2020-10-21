#--------------------------data-----------------------
# transform = torchvision.transforms.Compose(
#     [torchvision.transforms.Resize(size=(input_size, input_size), interpolation=Image.BICUBIC),
#      torchvision.transforms.ToTensor(),#Img2Tensor
#      torchvision.transforms.Normalize(mean=[0.5], std=[0.5])# 取值范围(0,1)->(-1,1)
#      #torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)), #单通道改三通道
#      #torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)

#      ]
# )
# train_loader = torch.utils.data.DataLoader(
#     #dataset=torchvision.datasets.FashionMNIST('./data/', train=True, download=True, transform=transform),
#     #dataset=torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform),
#     dataset=torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transform),
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=gpu_mode,
#     drop_last=True
# )

#celeba
# transform = torchvision.transforms.Compose([
#         torchvision.transforms.CenterCrop(160),
#         torchvision.transforms.Resize(64),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
# data_dir = '/_yucheng/dataSet/celeba/'  # this path depends on your computer
# train_loader =  utils.load_celebA(data_dir, transform, batch_size, shuffle=True)

#face_3d
# transform = torchvision.transforms.Compose([
#         #torchvision.transforms.CenterCrop(160),
#         torchvision.transforms.Resize((64,64)),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
# path = '/_yucheng/dataSet/face3d//face3d'
# face3d_dataset = torchvision.datasets.ImageFolder(path, transform=transform)
# train_loader = torch.utils.data.DataLoader(face3d_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

#-------------moving-mnist--------------
# train_set = utils.MovingMNIST(train=True,transform=torchvision.transforms.Normalize(mean=[127.5], std=[127.5]))#[0,255]->[-1,1]
# train_loader = torch.utils.data.DataLoader(
#                  dataset=train_set,
#                  batch_size=batch_size,
#                  shuffle=True,
#                  drop_last=True
#                  )


#nemo
# transform = torchvision.transforms.Compose([
#         #torchvision.transforms.CenterCrop(160),
#         torchvision.transforms.Resize((64,64)),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
# path = '/_yucheng/dataSet/nemo/nemo'
# face3d_dataset = torchvision.datasets.ImageFolder(path, transform=transform)
# train_loader = torch.utils.data.DataLoader(face3d_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

#------------ moco_actions----moco_shapes --------------
# transform = torchvision.transforms.Compose([
#         #torchvision.transforms.CenterCrop(160),
#         torchvision.transforms.Resize((64,64)),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
# path = '/_yucheng/dataSet/moco/moco_actions/'
# #path = '/_yucheng/dataSet/moco/moco_shapes/'
# face3d_dataset = torchvision.datasets.ImageFolder(path, transform=transform)
# train_loader = torch.utils.data.DataLoader(face3d_dataset, batch_size=batch_size, shuffle=True,drop_last=True)