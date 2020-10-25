import mindspore.dataset as ds
from optparse import OptionParser
import os
import PIL
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

import random
from mindspore.common import dtype as mstype
import mindspore.dataset.transforms.c_transforms as C
from RandAugment import RandAugment
# from utils.autoaugment import CIFAR10Policy, SVHNPolicy
# from utils.GaussianBlur import GaussianBlur

#import mindspore.dataset.transforms.vision.py_transforms as transforms
import numpy as np
from mindspore.dataset import GeneratorDataset
import pandas as pd
from collections import Counter
# v1.0
from mindspore.dataset.transforms.py_transforms import Compose
import mindspore.dataset.vision.py_transforms as transforms

# random.seed(1)
# np.random.seed(1)
# ds.config.set_seed(1)

# 数据集划分, 训练集:验证集:测试集 = 4:1:5
def split_train_val_test(sids):
    np.random.seed(286501567)
    np.random.shuffle(sids)
    ts = int(len(sids) * 0.4)
    vs = int(len(sids) * 0.5)

    return sids[:ts], sids[ts:vs], sids[vs:]

class TransformOnImg:
    def __init__(self, mode):
        self.mode = mode
        rand_augment = RandAugment(n=2,m=10)
        #v1.0
        self.trsfm_basic = Compose([
        #self.trsfm_basic = transforms.Compose([
            #transforms.Decode(),
            transforms.ToPIL(),
            transforms.Resize(256),
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomColorAdjust(0.4, 0.4, 0.4, 0.4),
            #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(prob=0.2),
            #transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.trsfm_aux = Compose([
        #self.trsfm_aux = transforms.Compose([
            #transforms.Decode(),
            transforms.ToPIL(),
            transforms.Resize(256),
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            rand_augment,
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.trsfm = trsfm = Compose([
        #self.trsfm = trsfm = transforms.Compose([
            #transforms.Decode(),
            transforms.ToPIL(),
            transforms.Resize(256),
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    def __call__(self, img, use_aux=False):
        if self.mode == "pretrain":
            if use_aux:
                img = self.trsfm_aux(img)
            else:
                img = self.trsfm_basic(img)
        else:
            img = self.trsfm_basic(img)
        return img

class BagDataCollatePretrain():
    def __init__(self):
        pass

    def __call__(self, batch):
        imgs_basic1, imgs_basic2, imgs_aux, anns = batch
        print("pretrain shape:",imgs_basic1.shape,imgs_basic2.shape,imgs_aux.shape, anns.shape)
        return imgs_basic1, imgs_basic2, imgs_aux, anns

class BagDataCollate():

    def __init__(self):
        pass

    def __call__(self, batch):
        # 输入一个batch的bag

        bsid, bimgs, blabel = batch
        size = len(bsid)

        # 统计每个bag的patch数量
        nslice = [x.shape[0] for x in bimgs]
        max_slice = max(nslice)

        # 通过拼接的方式将所有bag的patch数量统一为最大量
        pad_imgs = []
        for i in range(size):
            pad_img = np.pad(
                bimgs[i],
                [(0, max_slice - nslice[i]), (0, 0), (0, 0), (0, 0)],
                mode='constant',
                constant_values=0
            )
            pad_imgs.append(pad_img)

        # 将一个batch里面的bag根据patch的数量重新排序,使得更加的均衡
        nslice = np.array(nslice)
        order = balance_split(nslice)

        # 对这个batch的数据样本进行重排序
        bsid = np.array(bsid)[order]
        pad_imgs = np.array(pad_imgs)[order]
        blabel = np.array(blabel)[order]
        nslice = nslice[order]
        print("not pretrain shape:",np.array(bsid).shape,np.array(pad_imgs).shape, np.array(blabel).shape, np.array(nslice).shape)
        # 返回数据
        return np.array(bsid), np.array(pad_imgs), np.array(blabel), np.array(nslice)

class HPADataset:
    def __init__(self, data_dir, mode, batch_size, bag_size=20, classes=10):
        self.collate = BagDataCollate()
        self.collate_pretrain = BagDataCollatePretrain()
        self.nclasses = classes
        self.transform = TransformOnImg(mode = mode)
        self.mode = mode
        self.data_dir = data_dir
        self.bag_size = bag_size
        self.batch_size = batch_size
        filter_d, self.top_cv = self.filter_top_cv(classes)
        sids = np.array(list(sorted(filter_d.keys())))
        train_sids, val_sids, test_sids = split_train_val_test(sids)

        if mode == 'pretrain':
            self.db, self.sids = self.load_data(filter_d, train_sids, max_bag_size=1)
        elif mode == 'train':
            self.db, self.sids = self.load_data(filter_d, train_sids, max_bag_size=1)
        elif mode == 'val':
            self.db, self.sids = self.load_data(filter_d, val_sids, max_bag_size=20)
        elif mode == 'test':
            self.db, self.sids = self.load_data(filter_d, test_sids, max_bag_size=20)

        
    def load_data(self, d, sids, max_bag_size):
        '''
            最终的字典格式为:
            {
                "蛋白名_序号值":{
                    img: ["686_A3_2_blue_red_green_1.jpg", "686_A3_2_blue_red_green_2.jpg", ...]
                    label: ['2', '5', '6']
                }
                ...
            }
        '''
        # 制作字典
        imgdir = self.data_dir
        final_d = {}
        for sid in sids:
            # 读取这个目录下面的所有图片
            gene_imgs = []
            for gene_img in os.listdir(os.path.join(imgdir, sid)):
                img_pth = os.path.join(imgdir, sid, gene_img)
                gene_imgs.append(img_pth)

            # 对多于max_bag_size张的进行划分，确定每包包含的图片的大小
            gene_imgs = list(set(gene_imgs))
            bag_size = len(gene_imgs)
            while (bag_size > max_bag_size):
                bag_size = bag_size // 2

            # 保存满包的数据
            num_bags = len(gene_imgs) // bag_size
            for i in range(num_bags):
                bag_img = gene_imgs[i * bag_size: (i + 1) * bag_size]
                gene_name = '%s_%d' % (sid, i)

                final_d[gene_name] = {}
                final_d[gene_name]['img'] = bag_img
                final_d[gene_name]['label'] = d[sid]

            # 保存不满包的数据
            if (len(gene_imgs) > num_bags * bag_size):
                bag_img = gene_imgs[num_bags * bag_size:]
                gene_name = '%s_%d' % (sid, num_bags)

                final_d[gene_name] = {}
                final_d[gene_name]['img'] = bag_img
                final_d[gene_name]['label'] = d[sid]

        return final_d, sorted(final_d.keys())

    def get_sid_label(self, sid):
        sid_anns = self.db[sid]['label']
        anns = np.zeros(self.nclasses)
        for ann in sid_anns:
            anns[self.top_cv.index(ann)] = 1
        return anns


    def filter_top_cv(self, k=10, csv_file="enhanced.csv"):
        # 获取所有蛋白质文件夹的label
        all_cv = []
        label_file = pd.read_csv(csv_file)
        labels = label_file['label']
        for label in labels:
            all_cv += list(label.split(";"))

        # 统计label的个数,并获取top的label
        count = Counter(all_cv)
        top_cv = [x[0] for x in count.most_common(k)]

        # 首先制作目前的蛋白质文件夹和label对应的字典
        d = {}
        genes = label_file['Gene']
        labels = label_file['label']
        for i in range(len(genes)):
            d[genes[i]] = list(labels[i].split(";"))

        # 制作一个蛋白质文件夹字典，保存过滤后的对应关系
        filter_d = {}
        all_sids = sorted(d.keys())
        for sid in all_sids:
            # 保存top k的蛋白质文件夹的label
            for label in d[sid]:
                if label not in top_cv:
                    continue
                if sid not in filter_d:
                    filter_d[sid] = []
                filter_d[sid].append(label)

        # 返回过滤后的数据
        if len(top_cv) < k:
            print("Error: top cv less than k", count)
        return filter_d, top_cv


    def __len__(self):
        
        return len(self.sids) // self.batch_size

    def __getitem__(self, index):
        if self.mode == "pretrain":
            imgs_basic1 = []
            imgs_basic2 = []
            imgs_aux = []
            anns = []
            for idx in range(index*self.batch_size, (index+1)*self.batch_size):
                sid = self.sids[idx]
                sid_imgs = self.db[sid]['img']
                ann = self.get_sid_label(sid)
                for imgpth in sid_imgs:
                    img = Image.open(imgpth).convert('RGB')
                    img = np.asarray(img)
                    img_basic1 = self.transform(img)
                    img_basic2 = self.transform(img)
                    imgs_basic1.append(img_basic1)
                    imgs_basic2.append(img_basic2)
                    img_aux = self.transform(img,use_aux=True)
                    imgs_aux.append(img_aux)
                    anns.append(ann)

            imgs_basic1 = np.stack(imgs_basic1).astype(np.float)
            imgs_basic2 = np.stack(imgs_basic2).astype(np.float)
            imgs_aux = np.stack(imgs_aux).astype(np.float)
            anns = np.stack(anns).astype(np.int32)
            batch = (imgs_basic1, imgs_basic2, imgs_aux, anns)
            return self.collate_pretrain(batch)

        else:
            imgs = []
            anns = []
            sids = []
            for idx in range(index*self.batch_size, (index+1)*self.batch_size):
                sid = self.sids[idx]
                sid_imgs = self.db[sid]['img']
                ann = self.get_sid_label(sid)
                anns.append(ann)
                sids.append(sid)
                for imgpth in sid_imgs:
                    img = Image.open(imgpth).convert('RGB')
                    img = np.asarray(img)
                    img = self.transform(img)
                    imgs.append(img)
            print("imgs",imgs)
            print("sids",sids)
            print("anns",anns)

            # imgs = np.stack(imgs).astype(np.float)
            # anns = np.stack(anns).astype(np.int32)
            
            batch = (sids, imgs, anns)
            return self.collate(batch)


        
        # sid = self.sids[idx]
        # sid_imgs = self.db[sid]['img']
        # # 读取一个包内的所有图片，进行数据增强
        # imgs_basic1 = []
        # imgs_basic2 = []
        # imgs_aux = []
        # imgs = []
        # for imgpth in sid_imgs:
        #     img = Image.open(imgpth).convert('RGB')
        #     #img = Image.open(imgpth)
        #     img = np.asarray(img)
        #     if self.mode == 'pretrain':
        #         img_basic1 = self.transform(img)
                
        #         img_basic2 = self.transform(img)
                
        #         imgs_basic1.append(img_basic1)
                
        #         imgs_basic2.append(img_basic2)
        #         img_aux = self.transform(img,use_aux=True)
        #         imgs_aux.append(img_aux)
        #     else:
        #         img = self.transform(img)
        #         imgs.append(img)

        # anns = self.get_sid_label(sid)
        # if self.mode == 'pretrain':
        #     # np.random.shuffle(imgs_basic1)
        #     # np.random.shuffle(imgs_basic2)
        #     # np.random.shuffle(imgs_aux)
        #     imgs_basic1 = np.stack(imgs_basic1).astype(np.float)
        #     imgs_basic2 = np.stack(imgs_basic2).astype(np.float)
        #     imgs_aux = np.stack(imgs_aux).astype(np.float)
        #     return imgs_basic1, imgs_basic2, imgs_aux, anns
        # else:
        #     imgs = np.stack(imgs)
        #     return sid, imgs, anns


def makeup_pretrain_dataset(data_dir, batch_size, bag_size, epoch):

    pretrain_dataset = HPADataset(data_dir=data_dir, mode="pretrain", batch_size=batch_size, bag_size=bag_size)
    ds = GeneratorDataset(pretrain_dataset, ['img_basic1','img_basic2','img_aux','label'])
    #ds = ds.batch(batch_size)
    ds = ds.repeat(epoch)

    return ds

def makeup_dataset(data_dir, mode, batch_size, bag_size, epoch):

    dataset = HPADataset(data_dir=data_dir, mode=mode, batch_size=batch_size, bag_size=bag_size)
    ds = GeneratorDataset(dataset, ['sid','imgs','labels','nslices'])
    #ds = ds.batch(batch_size)
    ds = ds.repeat(epoch)

    return ds

# class CIFAR10Dataset():
#     def __init__(self, data_dir, training=True, use_third_trsfm=False, use_auto_augment=False, num_parallel_workers=8,
#                  device_num=1, device_id="0"):

#         if not training:
#             trsfm = transforms.ComposeOp([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             ])
#         else:
#             if not use_third_trsfm:
#                 trsfm = transforms.ComposeOp([
#                     transforms.ToPIL(),
#                     transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
#                     transforms.RandomColorAdjust(0.4, 0.4, 0.4, 0.4),
#                     transforms.RandomGrayscale(prob=0.2),
#                     transforms.RandomHorizontalFlip(),
#                     # GaussianBlur(kernel_size=int(0.1 * 32)),
#                     # GaussianBlur(),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                 ])
#             else:
#                 if use_auto_augment:
#                     trsfm = transforms.ComposeOp([
#                         transforms.ToPIL(),
#                         transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
#                         transforms.RandomHorizontalFlip(),
#                         CIFAR10Policy(),
#                         transforms.ToTensor(),
#                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                     ])
#                 else:
#                     rand_augment = RandAugment(n=2, m=10)
#                     trsfm = transforms.ComposeOp([
#                         transforms.ToPIL(),
#                         transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
#                         transforms.RandomHorizontalFlip(),
#                         rand_augment,
#                         transforms.ToTensor(),
#                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                     ])

#         self.trsfm = trsfm
#         self.data_dir = data_dir
#         self.num_parallel_workers = num_parallel_workers
#         self.device_num = device_num
#         self.device_id = device_id

#     def get_dataset(self):
#         print("get data from dir:", self.data_dir)
#         if self.device_num == 1:
#             ds_ = ds.Cifar10Dataset(self.data_dir, num_parallel_workers=self.num_parallel_workers)
#         else:
#             ds_ = ds.Cifar10Dataset(self.data_dir, num_parallel_workers=self.num_parallel_workers,
#                                     num_shards=self.device_num, shard_id=self.device_id)

#         ds_ = ds_.map(input_columns=["image"], operations=self.trsfm())
#         typecast_op = C.TypeCast(mstype.int32)
#         ds_ = ds_.map(input_columns="label", operations=typecast_op)
#         return ds_


# def makeup_train_dataset(ds1, ds2, ds3, batchsize, epoch):
#     ds1 = ds1.rename(input_columns=["label", "image"], output_columns=["label1", "data1"])
#     ds2 = ds2.rename(input_columns=["label", "image"], output_columns=["label2", "data2"])
#     ds3 = ds3.rename(input_columns=["image"], output_columns=["data3"])
#     ds_new = ds.zip((ds1, ds2))
#     ds_new = ds_new.project(columns=['data1', 'data2'])
#     ds_new = ds.zip((ds3, ds_new))
#     ds_new = ds_new.map(input_columns=['label'], output_columns=['label'],
#                         columns_order=['data3', 'data2', 'data1', 'label'],
#                         operations=lambda x: x)
#     # to keep the order : data3 data2 data1 label

#     # ds_new = ds_new.shuffle(ds_new.get_dataset_size())
#     print("dataset batchsize:",batchsize)
#     ds_new = ds_new.batch(batchsize)
#     ds_new = ds_new.repeat(epoch)

#     print("batch_size:", ds_new.get_batch_size(), "batch_num:", ds_new.get_dataset_size())

#     # for data in ds_new.create_dict_iterator():
#     #     print("new dataset:")
#     #     print(data.keys())
#     #     for key,value in data.items():
#     #         print("key:",key)
#     #
#     #     break

#     return ds_new


# def makeup_test_dataset(ds_test, batchsize, epoch=1):
#     ds_test = ds_test.batch(batchsize)
#     ds_test = ds_test.repeat(epoch)

#     return ds_test


# def get_train_dataset(train_data_dir, batchsize, epoch, mode="training", device_num=1, device_id="0"):
#     if mode == "linear_eval":
#         cifar10_train = CIFAR10Dataset(data_dir=train_data_dir, training=False, use_third_trsfm=False)
#         cifar10_train = cifar10_train.get_dataset()
#         cifar10_train = makeup_test_dataset(cifar10_train, batchsize, epoch)
#         return cifar10_train

#     cifar10_train_1 = CIFAR10Dataset(data_dir=train_data_dir, training=True, use_third_trsfm=False,
#                                      device_num=device_num, device_id=device_id)
#     cifar10_train_2 = CIFAR10Dataset(data_dir=train_data_dir, training=True, use_third_trsfm=False,
#                                      device_num=device_num, device_id=device_id)
#     cifar10_train_3 = CIFAR10Dataset(data_dir=train_data_dir, training=True, use_third_trsfm=True,
#                                      use_auto_augment=False, device_num=device_num, device_id=device_id)
#     cifar10_train_dataset1 = cifar10_train_1.get_dataset()
#     cifar10_train_dataset2 = cifar10_train_2.get_dataset()
#     cifar10_train_dataset3 = cifar10_train_3.get_dataset()
#     cifar10_train_dataset = makeup_train_dataset(cifar10_train_dataset1, cifar10_train_dataset2, cifar10_train_dataset3,
#                                                  batchsize=batchsize, epoch=epoch)

#     return cifar10_train_dataset


# def get_test_dataset(test_data_dir, batchsize, epoch=1, device_num=1, device_id="0"):
#     cifar10_test = CIFAR10Dataset(data_dir=test_data_dir, training=False, use_third_trsfm=False,
#                                   device_num=device_num, device_id=device_id)
#     cifar10_test_dataset = cifar10_test.get_dataset()
#     cifar10_test_dataset = makeup_test_dataset(cifar10_test_dataset, batchsize=batchsize, epoch=epoch)

#     return cifar10_test_dataset


# def get_train_test_dataset(train_data_dir, test_data_dir, batchsize, epoch=1):
#     cifar10_test = CIFAR10Dataset(data_dir=test_data_dir, training=False, use_third_trsfm=False)
#     cifar10_train = CIFAR10Dataset(data_dir=train_data_dir, training=False, use_third_trsfm=False)

#     cifar10_test_dataset = cifar10_test.get_dataset()
#     cifar10_train_dataset = cifar10_train.get_dataset()

#     func0 = lambda x, y: (x, y, np.array(0, dtype=np.int32))
#     func1 = lambda x, y: (x, y, np.array(1, dtype=np.int32))
#     input_cols = ["image", "label"]
#     output_cols = ["image", "label", "training"]
#     cols_order = ["image", "label", "training"]
#     cifar10_test_dataset = cifar10_test_dataset.map(input_columns=input_cols, output_columns=output_cols,
#                                                     operations=func0, columns_order=cols_order)
#     cifar10_train_dataset = cifar10_train_dataset.map(input_columns=input_cols, output_columns=output_cols,
#                                                       operations=func1, columns_order=cols_order)
#     # cifar10_train_dataset = cifar10_train_dataset.shuffle(cifar10_train_dataset.get_dataset_size())
#     # cifar10_test_dataset = cifar10_test_dataset.shuffle(cifar10_test_dataset.get_dataset_size())
#     concat_dataset = cifar10_train_dataset + cifar10_test_dataset
#     concat_dataset = concat_dataset.batch(batchsize)
#     concat_dataset = concat_dataset.repeat(epoch)

#     return concat_dataset


# if __name__ == "__main__":
#     TRAIN_DATA_DIR = "/home/tuyanlun/code/ms_r0.5/project/cifar-10-batches-bin/train"
#     train_dataset = get_train_dataset(TRAIN_DATA_DIR,128,200)
#     for data in train_dataset.create_dict_iterator():
#         print(data['data1'].shape) # (128,3,32,32)
#         print(data['data2'].shape) # (128,3,32,32)
#         print(data['data3'].shape) # (128,3,32,32)
#         print(data['label'].shape) # (128,)
#         print(data.keys())
#         break


if __name__ == "__main__":
    '''环境变量参数'''
    use_moxing = False
    if use_moxing:
        import moxing as mox

        # define local data path
        local_data_path = '/cache/data'

        mox.file.copy_parallel(src_url='s3://tuyanlun/data/', dst_url=local_data_path)
        # img = PIL.Image.open(os.path.join(local_data_path, 'GUI.png'))
        # print(img)
        TRAIN_DATA_DIR = os.path.join(local_data_path, "cifar10/cifar-10-batches-bin/train")
        TEST_DATA_DIR = os.path.join(local_data_path, "cifar10/cifar-10-batches-bin/test")
    else:
        # TRAIN_DATA_DIR = '/home/tuyanlun/code/ms_r0.5/project/cifar-10-batches-bin/train'
        # TEST_DATA_DIR = '/home/tuyanlun/code/ms_r0.5/project/cifar-10-batches-bin/test'
        DATA_DIR = '../hpa_dataset/hpa'

    hpa_pretrain_dataset = makeup_pretrain_dataset(data_dir=DATA_DIR,batch_size=64, bag_size=1, epoch=100)  
    # cifar10_train_dataset = get_train_dataset(TRAIN_DATA_DIR, 128, 200)
    # get_train_test_dataset(TRAIN_DATA_DIR, TEST_DATA_DIR, 128, 1)

    # for data in hpa_pretrain_dataset.create_dict_iterator():
    #     print(data.keys())
    #     print(data)
    #     break

    ds = hpa_pretrain_dataset.create_dict_iterator()
    data=ds.get_next()
    print("pretrain data",data)
    hpa_train_dataset = makeup_dataset(data_dir=DATA_DIR,mode='train',batch_size=64,bag_size=1,epoch=20)
    hpa_test_dataset = makeup_dataset(data_dir=DATA_DIR,mode='test',batch_size=3,bag_size=20,epoch=20)
    hpa_val_dataset = makeup_dataset(data_dir=DATA_DIR,mode='val',batch_size=3,bag_size=20,epoch=20)
    ds = hpa_train_dataset.create_dict_iterator()
    data=ds.get_next()
    print("train data",data)
    ds = hpa_val_dataset.create_dict_iterator()
    data=ds.get_next()
    print("val data",data)
    ds = hpa_test_dataset.create_dict_iterator()
    data=ds.get_next()
    print("test data",data)

    # print(TRAIN_DATA_DIR)
    # cifar10_train_dataset = get_train_dataset(TRAIN_DATA_DIR, 128, 200)
    #
    # def inverse_normal(data):
    #     '''
    #     Args:
    #         data: np.ndarray
    #     Returns:
    #     '''
    #     print(data.shape,type(data))
    #     mean = [0.4914, 0.4822, 0.4465]
    #     std = [0.2023, 0.1994, 0.2010]
    #     for i in range(len(mean)): # 反标准化
    #         data[i] = data[i] * std[i] + mean[i]
    #     data = data * 255
    #     data = data.astype(np.uint8)
    #     data = np.transpose(data,(1,2,0)) # (ch,h,w) -> (h,w,ch)
    #     return data
    #
    # def plot_img(data):
    #     '''
    #
    #     Args:
    #         data: np.ndarray
    #
    #     '''
    #     data = Image.fromarray(data,mode='RGB')
    #     print(data)
    #     #data.show()
    #     data.save('test.jpg')
    #
    #
    # for data in cifar10_train_dataset.create_dict_iterator():
    #     print(data['data1'].shape)
    #     t=inverse_normal(data['data1'][0])
    #     print(type(t),t)
    #     plot_img(t)
    #     #print(data['data2'])
    #     #print(data['data3'])
    #     print(data['label'])
    #     break
