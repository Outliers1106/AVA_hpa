import mindspore.dataset as ds
from optparse import OptionParser
import os
import PIL
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

import random
from mindspore.common import dtype as mstype
import mindspore.dataset.transforms.c_transforms as C
from src.RandAugment import RandAugment
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
        self.trsfm = Compose([
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
            img = self.trsfm(img)
        return img

class BagDataCollatePretrain():
    def __init__(self):
        pass

    def __call__(self, batch):
        imgs_basic1, imgs_basic2, imgs_aux, anns = batch
        #print("pretrain shape:",imgs_basic1.shape,imgs_basic2.shape,imgs_aux.shape, anns.shape)
        return imgs_basic1, imgs_basic2, imgs_aux, anns

class BagDataCollate():

    def __init__(self,mode):
        self.mode = mode

    def aggregate(self, img, label, nslice):
        nb, _, c, h, w = img.shape
        _, nclasses = label.shape
        img_count = np.sum(nslice)
        allimgs = np.zeros([img_count, c, h, w])
        alllabels = np.zeros([img_count, nclasses])
        cur = 0
        for s in range(nb):
            allimgs[cur:cur + nslice[s]] = img[s, :nslice[s], :]
            alllabels[cur:cur + nslice[s]] = label[s, :]
            cur = cur + nslice[s]
        
        if self.mode == "train":
            return allimgs.astype(np.float32), alllabels.astype(np.float32)
        else: # need `nslice` to recover bag when eval
            return allimgs.astype(np.float32), label.astype(np.float32), nslice.astype(np.float32)

        

    def __call__(self, batch):
        # 输入一个batch的bag

        bsid, bimgs, blabel = batch
        # print("bsid",bsid)
        # print("bimgs",bimgs)
        # print("blabel",blabel)
        size = len(bsid)
        
        # 统计每个bag的patch数量
        nslice = [x.shape[0] for x in bimgs]
        max_slice = max(nslice)
        # print("point--------point")
        # print("nslice",nslice)
        # print("size",size)
        # print("max_slice",max_slice)
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
        #print("not pretrain shape:",np.array(bsid).shape,np.array(pad_imgs).shape, np.array(blabel).shape, np.array(nslice).shape)
        # 返回数据
        print(np.array(pad_imgs).shape, np.array(blabel).shape, np.array(nslice).shape)
        #TODO 返回的得是The type of `tensor input_data` should be one of ['Tensor', 'float', 'int'], but got ndarray.
        return self.aggregate(np.array(pad_imgs), np.array(blabel),np.array(nslice))
        # return np.array(pad_imgs), np.array(blabel)
        #return np.array(pad_imgs),np.array(pad_imgs),np.array(pad_imgs),np.array(pad_imgs)
# 均衡操作
def find_i_j_v(seq):
    '''isOk[i][j][v]: find j numbers from front i sum to v
    for seq, index starts from 0
    for isOk mark, index starts from 1
    '''
    n = len(seq)
    tot = np.sum(seq)

    isOk = np.zeros((n + 1, n + 1, tot + 1), dtype=int)
    isOk[:, 0, 0] = 1

    for i in range(1, n + 1):
        jmax = min(i, n // 2)
        for j in range(1, jmax + 1):
            for v in range(1, tot // 2 + 1):
                if isOk[i - 1][j][v]:
                    isOk[i][j][v] = 1

            for v in range(1, tot // 2 + 1):
                if v >= seq[i - 1]:
                    if isOk[i - 1][j - 1][v - seq[i - 1]]:
                        isOk[i][j][v] = 1
    return isOk


def balance_split(seq):
    '''split seq to 2 sub list with equal length, sum nearly equal '''
    n = len(seq)
    tot = np.sum(seq)
    res = find_i_j_v(seq)

    i = n
    j = n // 2
    v = tot // 2

    sel_idx = []
    sel_val = []

    while not res[i][j][v] and v > 0:
        v = v - 1

    while len(sel_idx) < n // 2 and i >= 0:
        if res[i][j][v] and res[i - 1][j - 1][v - seq[i - 1]]:
            sel_idx.append(i - 1)
            sel_val.append(seq[i - 1])
            j = j - 1
            v = v - seq[i - 1]
            i = i - 1
        else:
            i = i - 1

    left = sel_idx
    right = [x for x in list(range(n)) if x not in left]
    return np.array(left + right)


class HPADataset:
    def __init__(self, data_dir, mode, batch_size, bag_size=20, classes=10):
        self.collate = BagDataCollate(mode=mode)
        self.collate_pretrain = BagDataCollatePretrain()
        self.nclasses = classes
        self.transform = TransformOnImg(mode = mode)
        self.mode = mode
        self.data_dir = data_dir
        self.bag_size = bag_size
        self.batch_size = batch_size
        filter_d, self.top_cv = self.filter_top_cv(classes)
        # print("len(filter_d:{}".format(len(filter_d)))
        # print("self.top_Cv:{}".format(self.top_cv))
        sids = np.array(list(sorted(filter_d.keys())))
        train_sids, val_sids, test_sids = split_train_val_test(sids)

        if mode == 'pretrain':
            self.db, self.sids = self.load_data(filter_d, train_sids, max_bag_size=1) # max_bag_size=1
        elif mode == 'train':
            self.db, self.sids = self.load_data(filter_d, train_sids, max_bag_size=1) # max_bag_size=1
        elif mode == 'val':
            self.db, self.sids = self.load_data(filter_d, val_sids, max_bag_size=20) # max_bag_size=20
        elif mode == 'test':
            self.db, self.sids = self.load_data(filter_d, test_sids, max_bag_size=20) # max_bag_size=20

        
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
            # print("get item: pretrain")
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

            imgs_basic1 = np.stack(imgs_basic1).astype(np.float32)
            imgs_basic2 = np.stack(imgs_basic2).astype(np.float32)
            imgs_aux = np.stack(imgs_aux).astype(np.float32)
            anns = np.stack(anns).astype(np.int32)
            batch = (imgs_basic1, imgs_basic2, imgs_aux, anns)
            return self.collate_pretrain(batch)

        else:
            # print("get item: not pretrain")
            imgs_tuple=[]
            anns_tuple=[]
            sids_tuple=[]
            
            for idx in range(index*self.batch_size, (index+1)*self.batch_size):
                imgs = []

                sid = self.sids[idx]
                #print("sid:",sid)
                sid_imgs = self.db[sid]['img']
                #print("sid imgs:",sid_imgs)
                ann = self.get_sid_label(sid)
                #print("ann:",ann)
                
                for imgpth in sid_imgs:
                    img = Image.open(imgpth).convert('RGB')
                    img = np.asarray(img)
                    img = self.transform(img)
                    imgs.append(img)
                
                imgs = np.stack(imgs)
                imgs_tuple.append(imgs)
                anns_tuple.append(ann)
                sids_tuple.append(sid)
                
            batch = (tuple(sids_tuple), tuple(imgs_tuple), tuple(anns_tuple))
            return self.collate(batch)



def makeup_pretrain_dataset(data_dir, batch_size, bag_size, epoch=1):

    pretrain_dataset = HPADataset(data_dir=data_dir, mode="pretrain", batch_size=batch_size, bag_size=bag_size)
    ds = GeneratorDataset(pretrain_dataset, ['img_basic1','img_basic2','img_aux','label'])
    #ds = ds.batch(batch_size)
    ds = ds.repeat(epoch)

    return ds

def makeup_dataset(data_dir, mode, batch_size, bag_size, epoch=1):

    dataset = HPADataset(data_dir=data_dir, mode=mode, batch_size=batch_size, bag_size=bag_size)
    ds = GeneratorDataset(dataset, ['imgs','labels'])
    #ds = ds.batch(batch_size)
    ds = ds.repeat(epoch)

    return ds




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
    ds = hpa_pretrain_dataset.create_dict_iterator()
    for data in ds:
        print(data)
    # data=ds.get_next()
    # print("pretrain data",data)
    # hpa_train_dataset = makeup_dataset(data_dir=DATA_DIR,mode='train',batch_size=64,bag_size=1,epoch=20)
    # ds = hpa_train_dataset.create_dict_iterator()
    # data=ds.get_next()
    # print("train data",data)
    # hpa_val_dataset = makeup_dataset(data_dir=DATA_DIR,mode='test',batch_size=3,bag_size=20,epoch=20)
    # ds = hpa_val_dataset.create_dict_iterator()
    # data=ds.get_next()
    # print("val data",data)
    # hpa_test_dataset = makeup_dataset(data_dir=DATA_DIR,mode='val',batch_size=3,bag_size=20,epoch=20)
    # ds = hpa_test_dataset.create_dict_iterator()
    # data=ds.get_next()
    # print("test data",data)
