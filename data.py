# coding=utf-8
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class DatasetFromFolder(Dataset):
    def __init__(self,path='',transform=None, channels = 3):
        super().__init__()
        self.channels = channels
        self.path = path
        self.transform = transform
        self.image_filenames = [x for x in os.listdir(self.path) if x.endswith('jpg') or x.endswith('png')] # x.startswith() 
        #imgs_path = os.listdir(path)
        #self.image_filenames = list(filter(lambda x:x.endswith('jpg') or x.endswith('png') ,imgs_path))
    def __getitem__(self, index):
        if self.channels == 1:
            a = Image.open(os.path.join(self.path, self.image_filenames[index])).convert('L') # 'L'是灰度图, 'RGB'彩色
        elif self.channels ==3:
            a = Image.open(os.path.join(self.path, self.image_filenames[index])).convert('RGB')
        else:
            print('error')
        #a = a.resize((self.size, self.size), Image.BICUBIC)
        #a = transforms.ToTensor()(a)
        if self.transform:
            a = self.transform(a)
        return a
    def __len__(self):
        return len(self.image_filenames)

def make_dataset(dataset_name, batch_size,img_size,drop_remainder=True, shuffle=True, num_workers=4, pin_memory=False,img_paths=''):
    if dataset_name == 'mnist' or dataset_name=='fashion_mnist':
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        if dataset_name == 'mnist':
            dataset = datasets.MNIST('data/MNIST', transform=transform, download=True)
        else:
            dataset = datasets.FashionMNIST('data/FashionMNIST', transform=transform, download=True)
        img_shape = [img_size, img_size, 1]
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = datasets.CIFAR10('data/CIFAR10', transform=transform, download=True)
        img_shape = [32, 32, 3]
    elif dataset_name == 'pose10':
        transform_pose10 = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5], std=[0.5]) #黑白应该是不用norm
        ])
        #dataset = DatasetFromFolder(path='/_yucheng/dataSet/pose/pose_set_10',size=img_size)
        #dataset = DatasetFromFolder(path='./data/Pose/pose_set_10',size=img_size)
        path_pose10='./data/Pose/pose_set_10'
        dataset = DatasetFromFolder(path=path_pose10,transform=transform_pose10,channels=1)
        img_shape = [img_size, img_size, 1]
    elif dataset_name == 'celeba_64':
        crop_size = 108
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]# [image,height,width]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(crop),
            transforms.Resize(size=(img_size, img_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            #transforms.ToPILImage()
            ])
        dataset = DatasetFromFolder(path='',size=64)
        img_shape = (img_size, img_size, 3)
    elif dataset_name == 'celeba_HQ':
        transform_128 = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        path_128 = 'F:/dataSet2/CelebAMask-HQ/CelebA-HQ-img'
        #path_128 = '/home/disanda/Desktop/dataSet/celeba-hq-download/celeba-128'
        #path_128 = '/_yucheng/dataSet/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img'
        dataset = DatasetFromFolder(path=path_128,transform=transform_128,channels=3)
        img_shape = (img_size, img_size, 3)
    else:
        raise NotImplementedError
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_remainder, pin_memory=pin_memory)
    return data_loader, img_shape