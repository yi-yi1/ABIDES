import os
import pandas as pd
import numpy as np
import numpy.ma as ma
import random
from sklearn.metrics import  confusion_matrix
import openpyxl
import torch.optim as optim

from Data_Augment import *

#e.g:  给定的文件名为Caltech_0051456_rois_cc200.1D
#      返回的key为Caltech_0051456
def get_key(filename):
    f_split = filename.split('_')
    if f_split[3] == 'rois':
        key = '_'.join(f_split[0:3])      #以
    else:
        key = '_'.join(f_split[0:2])
    return key

def get_key2(filename):
    f_split=filename.split('_')
    return f_split[1][7:12]


##给定文件名和labels，获得文件名对应的标签
# labels中存放的为每文件id对应的标签。
# e.g: Pitt_0050003': 1
def get_label(filename,labels):

     if filename in labels:
         return labels[filename]
     else:
         return 0

def get_label2(filename,labels):
    # if filename in labels:
    return labels[filename]
    # else:
    #     return 0

def get_time_series_data(filename,data_path):
    for file in os.listdir(data_path):
        if file.startswith(filename):
            df = pd.read_csv(os.path.join(data_path, file), sep='\t')
    return np.array(df)



#给定一个文件名（一个subject的时间序列数据），通过计算皮尔逊相关系数，
#得到单个subject的一维特征向量，共19990个特征
def get_corr_data(filename,data_main_path):
    # print(filename)
    for file in os.listdir(data_main_path):
        # startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False。
        if file.startswith(filename):
            # os.path.join() 函数用于路径拼接文件路径
            df = pd.read_csv(os.path.join(data_main_path, file), sep='\t')
            # print(df.T)
    # numpy.errstate（）：用于浮点错误处理的上下文管理器
    with np.errstate(invalid="ignore"):
        # numpy.nan_to_num(x):使用0代替数组x中的nan元素，使用有限的数字代替inf元素
        # numpy.corrcoef() ：返回皮尔逊积矩相关系数
        # df.T 求df的转置
        # 得到了单个文件(subject)的皮尔逊积矩相关系数
        corr = np.nan_to_num(np.corrcoef(df.T))
        # 函数invert()计算输入数组中整数的二进制按位NOT结果（取反）
        # numpy.tri（）函数返回一个值为1和0的数组,若dtype为bool，则返回的是值为true 和falase的数组。默认情况下行数与列数相等
        # 第三个参数是k，它是一个整数值，默认情况下为0。如果k> 0的值，则表示对角线位于主对角线上方，反之亦然。
        temp = np.tri(corr.shape[0], k=-1, dtype=bool)
        mask = np.invert(temp)
        # ma.masked_where（condition，a，copy = True ）
        # 屏蔽满足条件的数组。
        # 即m为mask不为1的数组
        m = ma.masked_where(mask == 1, mask)
        # 返回的是皮尔逊相关矩阵的上上三角，并且拉平为一维向量。
        return ma.masked_where(m, corr).compressed()


# 返回精确率、敏感性、特异性
#函数参数为标签值和预测值
def confusion(g_turth, predictions):
    # ravel()：返回一个展平的数组。
    tn, fp, fn, tp = confusion_matrix(g_turth, predictions, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    sensitivity = (tp) / (tp + fn)
    specificty = (tn) / (tn + fp)
    return accuracy, sensitivity, specificty


#切片数据增强,crops为切片数
def get_agument_data(all_data,sample_list,crops):
    data=None
    label=None
    for f in sample_list:
        max_seq_length = all_data[f][0].shape[0]
        if data is None:
                data = np.expand_dims(feature_normalize(all_data[f][0]),axis=0)
                print(data.shape)
                label=all_data[f][1]
        else:
                temp=np.expand_dims(feature_normalize(all_data[f][0]),axis=0)
                data = np.vstack((data,temp))
                print(data.shape)
                label=np.append(label,all_data[f][1])
    return data,label


#crops为切片的片数
def data_agument(all_data,sample_list,fix_seq_length,crops):
    augment_data=[]
    agument_label=[]

    for f in sample_list:
        max_seq_length=all_data[f][0].shape[0]

        range_list=list(range(fix_seq_length+1,int(max_seq_length)))

        random_index=random.sample(range_list,crops)

        for j in range(crops):
            r=random_index[j]
            augment_data.append(all_data[f][0][r-fix_seq_length:r])
            agument_label.append(all_data[f][1])
            # agument_label.append(all_data[f][1] - 1)

    print(np.array(augment_data).shape)
    print(np.array(agument_label).shape)
    # print(np.array(augment_data))
    # print(np.array(agument_label))

    return np.array(augment_data),np.array(agument_label)


def data_agument2(all_data,Pheno_Info,sample_list,fix_seq_length,crops):
    augment_data=[]
    agument_label=[]
    augmented_pheno=[]
    nor_augmented_pheno=[]



    for f in sample_list:
        max_seq_length=all_data[f][0].shape[0]

        range_list=list(range(fix_seq_length+1,int(max_seq_length)))

        random_index=random.sample(range_list,crops)

        for j in range(crops):
            r=random_index[j]
            augment_data.append(all_data[f][0][r-fix_seq_length:r])
            agument_label.append(all_data[f][1])
            augmented_pheno.append(Pheno_Info[f])

    augmented_pheno=np.array(augmented_pheno)
    for i in range(augmented_pheno.shape[0]):
        r_mean = np.sum(np.abs(augmented_pheno[i, :])) / augmented_pheno.shape[1]
        for j in range(augmented_pheno.shape[1]):
            if r_mean == 0:
                augmented_pheno[i, j] = 0
            else:
                augmented_pheno[i, j] = augmented_pheno[i, j] / r_mean
    print("data.shape",np.array(augment_data).shape)
    print("Pheno_Info.shape",np.array(augmented_pheno).shape)
    # print(augmented_pheno)

    return np.array(augment_data),np.array(agument_label),np.array(augmented_pheno)


#不同比例切片
def SubClass_data_agument(all_data,sample_list,fix_seq_length,crops):
    augment_data = []
    agument_label = []

    random.shuffle(sample_list)
    for f in sample_list:
        if all_data[f][1]==1:
            crops=10
        elif all_data[f][1] == 2:
            crops =23
        elif all_data[f][1]==3:
            crops=41
        max_seq_length = all_data[f][0].shape[0]
        range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
        random_index = random.sample(range_list, crops)
        for j in range(crops):
            r = random_index[j]
            augment_data.append(all_data[f][0][r - fix_seq_length:r])
            agument_label.append(all_data[f][1] - 1)

    print(np.array(augment_data).shape)
    print(np.array(agument_label).shape)

    return np.array(augment_data), np.array(agument_label)



#只加高斯噪声
def SubClass_data_agument3(all_data,sample_list,fix_seq_length,crops):
    augment_data = []
    agument_label = []
    flist_PDD_NOS=[]
    flist_Asperger=[]

    count1=0
    count2=0
    count3=0
    for f in  sample_list:
        if(all_data[f][1]==1):
            count1+=1
        elif(all_data[f][1]==2):
            count2+=1
            flist_Asperger.append(f)
        elif(all_data[f][1]==3):
            count3+=1
            flist_PDD_NOS.append(f)

    print("Count1",count1)
    print("Count2",count2)
    print("Count3",count3)

    #Asperger 数据增强
    random.shuffle(flist_Asperger)
    for f in flist_Asperger:
        temp=all_data[f][0]
        noise = np.random.standard_normal(size=(all_data[f][0].shape[0], all_data[f][0].shape[1]))
        temp=temp+noise
        label=all_data[f][1]
        max_seq_length =temp.shape[0]
        range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
        random_index = random.sample(range_list, crops)
        for j in range(crops):
            r = random_index[j]
            augment_data.append(temp[r - fix_seq_length:r])
            agument_label.append(label-1)

    count_Asperger=0
    random.shuffle(flist_Asperger)
    for f in flist_Asperger:
        if(count_Asperger<count1-2*count2):
            temp = all_data[f][0]
            noise = np.random.standard_normal(size=(all_data[f][0].shape[0], all_data[f][0].shape[1]))
            temp = temp + noise
            label = all_data[f][1]
            max_seq_length = temp.shape[0]
            range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
            random_index = random.sample(range_list, crops)
            for j in range(crops):
                r = random_index[j]
                augment_data.append(temp[r - fix_seq_length:r])
                agument_label.append(label - 1)
            count_Asperger+=1

    #PDD_NOS数据增强
    for i in range(3):
        random.shuffle(flist_PDD_NOS)
        for f in flist_PDD_NOS:
            temp=all_data[f][0]
            noise = np.random.standard_normal(size=(all_data[f][0].shape[0], all_data[f][0].shape[1]))
            temp=temp+noise
            label=all_data[f][1]
            max_seq_length =temp.shape[0]
            range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
            random_index = random.sample(range_list, crops)
            for j in range(crops):
                r = random_index[j]
                augment_data.append(temp[r - fix_seq_length:r])
                agument_label.append(label-1)
    count_PDD_NOS=0
    random.shuffle(flist_PDD_NOS)
    for f in flist_PDD_NOS:
        if(count_PDD_NOS<count1-4*count3):
            temp = all_data[f][0]
            noise = np.random.standard_normal(size=(all_data[f][0].shape[0], all_data[f][0].shape[1]))
            temp = temp + noise
            label = all_data[f][1]
            max_seq_length = temp.shape[0]
            range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
            random_index = random.sample(range_list, crops)
            for j in range(crops):
                r = random_index[j]
                augment_data.append(temp[r - fix_seq_length:r])
                agument_label.append(label - 1)
            count_PDD_NOS+=1


    # print(np.array(augment_data).shape)
    # print(np.array(agument_label).shape)

    random.shuffle(sample_list)
    for f in sample_list:
        max_seq_length = all_data[f][0].shape[0]
        range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
        random_index = random.sample(range_list, crops)
        for j in range(crops):
            r = random_index[j]
            augment_data.append(all_data[f][0][r - fix_seq_length:r])
            agument_label.append(all_data[f][1] - 1)

    print(np.array(augment_data).shape)
    print(np.array(agument_label).shape)

    return np.array(augment_data), np.array(agument_label)


# 高斯+GAN
def SubClass_data_agument4(all_data,sample_list,fix_seq_length,crops):
    augment_data = []
    agument_label = []
    flist_PDD_NOS=[]
    flist_Asperger=[]
    flist_Autism=[]

    data_Asperger=[]
    data_PDD_NOS=[]
    #
    # augment_data_Asperger=[]
    # agument_label_Asperger=[]
    #
    # augment_data_PDD_NOS = []
    # agument_label_PDD_NOS = []
    #
    augment_data_Autism = []
    agument_label_Autism = []

    count1=0
    count2=0
    count3=0
    for f in sample_list:
        if(all_data[f][1]==1):
            count1+=1
            flist_Autism.append(f)
        elif(all_data[f][1]==2):
            count2+=1
            flist_Asperger.append(f)
        elif(all_data[f][1]==3):
            count3+=1
            flist_PDD_NOS.append(f)

    # print("Count1",count1)
    # print("Count2",count2)
    # print("Count3",count3)

    #Asperger 数据增强
    random.shuffle(flist_Asperger)
    for f in flist_Asperger:
        temp=all_data[f][0]
        noise = np.random.standard_normal(size=(all_data[f][0].shape[0], all_data[f][0].shape[1]))
        temp=temp+noise
        label=all_data[f][1]
        max_seq_length =temp.shape[0]
        range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
        random_index = random.sample(range_list, crops)
        for j in range(crops):
            r = random_index[j]
            data_Asperger.append(temp[r - fix_seq_length:r])
            data_Asperger.append(all_data[f][0][r - fix_seq_length:r])
            # agument_label_Asperger.append(label-1)
            # agument_label_Asperger.append(label-1)

    data_Asperger=np.array(data_Asperger)
    print(data_Asperger.shape)
    generator_model = Generator()
    discriminator_model = Discriminator()
    optimizer_G = optim.Adam(generator_model.parameters(),lr=0.0001)
    optimizer_D = optim.Adam(discriminator_model.parameters(),lr=0.0001)
    data_Asperger=torch.from_numpy(data_Asperger)
    data_Asperger=data_Asperger.to(torch.float32)
    reshape_data_Asperger=data_Asperger.view(data_Asperger.size(0),-1)
    data_Asperger=data_Asperger.numpy()
    data_Asperger_label=np.ones(data_Asperger.shape[0])
    print(data_Asperger_label.shape)

    agument_data_Asperger=Gan_augment_train(reshape_data_Asperger, generator_model, discriminator_model,50, optimizer_G,optimizer_D,count=(count1-count2*2)*10)
    agument_data_Asperger=torch.reshape(agument_data_Asperger,(-1,60,200))
    print(agument_data_Asperger.shape)
    agument_data_Asperger=agument_data_Asperger.detach().numpy()
    agument_Asperger_label=np.ones(agument_data_Asperger.shape[0])
    print(agument_Asperger_label.shape)


    #PDD_NOS数据增强
    random.shuffle(flist_PDD_NOS)
    for f in flist_PDD_NOS:
        temp=all_data[f][0]
        noise = np.random.standard_normal(size=(all_data[f][0].shape[0], all_data[f][0].shape[1]))
        temp=temp+noise
        label=all_data[f][1]
        max_seq_length =temp.shape[0]
        range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
        random_index = random.sample(range_list, crops)
        for j in range(crops):
            r = random_index[j]
            data_PDD_NOS.append(temp[r - fix_seq_length:r])
            data_PDD_NOS.append(all_data[f][0][r - fix_seq_length:r])
    for i in range(2):
        for f in flist_PDD_NOS:
            temp = all_data[f][0]
            noise = np.random.standard_normal(size=(all_data[f][0].shape[0], all_data[f][0].shape[1]))
            temp = temp + noise
            label = all_data[f][1]
            max_seq_length = temp.shape[0]
            range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
            random_index = random.sample(range_list, crops)
            for j in range(crops):
                r = random_index[j]
                data_PDD_NOS.append(temp[r - fix_seq_length:r])
                # data_PDD_NOS.append(all_data[f][0][r - fix_seq_length:r])
    data_PDD_NOS = np.array(data_PDD_NOS)
    print(data_PDD_NOS.shape)
    generator_model = Generator()
    discriminator_model = Discriminator()
    optimizer_G = optim.Adam(generator_model.parameters(), lr=0.0001)
    optimizer_D = optim.Adam(discriminator_model.parameters(), lr=0.0001)
    data_PDD_NOS=torch.from_numpy(data_PDD_NOS)
    data_PDD_NOS=data_PDD_NOS.to(torch.float32)
    reshape_data_PDD_NOS=data_PDD_NOS.view(data_PDD_NOS.size(0), -1)
    data_PDD_NOS=data_PDD_NOS.numpy()
    data_PDD_NOS_label=np.ones(data_PDD_NOS.shape[0])
    data_PDD_NOS_label += 1
    print(data_PDD_NOS_label)
    print(data_PDD_NOS_label.shape)
    agument_data_PDD_NOS = Gan_augment_train2(reshape_data_PDD_NOS, generator_model, discriminator_model,50, optimizer_G,optimizer_D,count=(count1-4*count3)*10)
    agument_data_PDD_NOS = torch.reshape(agument_data_PDD_NOS, (-1, 60, 200))
    print(agument_data_PDD_NOS.shape)
    agument_data_PDD_NOS=agument_data_PDD_NOS.detach().numpy()
    agument_PDD_NOS_label = np.ones(agument_data_PDD_NOS.shape[0])
    agument_PDD_NOS_label+=1
    print(agument_PDD_NOS_label.shape)

    flist_Autism=np.array(flist_Autism)
    random.shuffle(flist_Autism)
    for f in flist_Autism:
        max_seq_length = all_data[f][0].shape[0]
        range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
        random_index = random.sample(range_list, crops)
        for j in range(crops):
            r = random_index[j]
            augment_data_Autism.append(all_data[f][0][r - fix_seq_length:r])
            agument_label_Autism.append(all_data[f][1] - 1)
    augment_data.extend(agument_data_Asperger)
    augment_data.extend(data_Asperger)
    augment_data.extend(agument_data_PDD_NOS)
    augment_data.extend(data_PDD_NOS)
    augment_data.extend(augment_data_Autism)
    agument_label.extend(agument_Asperger_label)
    agument_label.extend(data_Asperger_label)
    agument_label.extend(agument_PDD_NOS_label)
    agument_label.extend(data_PDD_NOS_label)
    agument_label.extend(agument_label_Autism)

    print(np.array(augment_data).shape)
    print(np.array(agument_label).shape)

    return np.array(augment_data), np.array(agument_label)



#原始数据GAN+高斯
def SubClass_data_agument5(all_data,sample_list,fix_seq_length,crops):
    augment_data = []
    agument_label = []
    flist_PDD_NOS=[]
    flist_Asperger=[]
    flist_Autism=[]

    data_Asperger=[]
    data_PDD_NOS=[]

    Gauss_data_Asperger=[]
    Gauss_data_PDD_NOS=[]
    #
    # augment_data_Asperger=[]
    # agument_label_Asperger=[]
    #
    # augment_data_PDD_NOS = []
    # agument_label_PDD_NOS = []
    #
    augment_data_Autism = []
    agument_label_Autism = []

    count1=0
    count2=0
    count3=0
    for f in sample_list:
        if(all_data[f][1]==1):
            count1+=1
            flist_Autism.append(f)
        elif(all_data[f][1]==2):
            count2+=1
            flist_Asperger.append(f)
        elif(all_data[f][1]==3):
            count3+=1
            flist_PDD_NOS.append(f)

    # print("Count1",count1)
    # print("Count2",count2)
    # print("Count3",count3)

    #Asperger 数据增强
    random.shuffle(flist_Asperger)
    for f in flist_Asperger:
        temp=all_data[f][0]
        noise = np.random.standard_normal(size=(all_data[f][0].shape[0], all_data[f][0].shape[1]))
        temp=temp+noise
        label=all_data[f][1]
        max_seq_length =temp.shape[0]
        range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
        random_index = random.sample(range_list, crops)
        for j in range(crops):
            r = random_index[j]
            # data_Asperger.append(temp[r - fix_seq_length:r])
            data_Asperger.append(all_data[f][0][r - fix_seq_length:r])
            # agument_label_Asperger.append(label-1)
            # agument_label_Asperger.append(label-1)

    data_Asperger=np.array(data_Asperger)
    print(data_Asperger.shape)
    generator_model = Generator()
    discriminator_model = Discriminator()
    optimizer_G = optim.Adam(generator_model.parameters(),lr=0.0001)
    optimizer_D = optim.Adam(discriminator_model.parameters(),lr=0.0001)
    data_Asperger=torch.from_numpy(data_Asperger)
    data_Asperger=data_Asperger.to(torch.float32)
    reshape_data_Asperger=data_Asperger.view(data_Asperger.size(0),-1)
    data_Asperger=data_Asperger.numpy()
    data_Asperger_label=np.ones(data_Asperger.shape[0])
    print(data_Asperger_label.shape)

    agument_data_Asperger=Gan_augment_train(reshape_data_Asperger,generator_model,discriminator_model,100,optimizer_G,optimizer_D,count=(count1-count2*2)*10)
    agument_data_Asperger=torch.reshape(agument_data_Asperger,(-1,60,200))
    print(agument_data_Asperger.shape)
    agument_data_Asperger=agument_data_Asperger.detach().numpy()
    agument_Asperger_label=np.ones(agument_data_Asperger.shape[0])
    print(agument_Asperger_label.shape)

    for f in flist_Asperger:
        temp = all_data[f][0]
        noise = np.random.standard_normal(size=(all_data[f][0].shape[0], all_data[f][0].shape[1]))
        temp = temp + noise
        label = all_data[f][1]
        max_seq_length = temp.shape[0]
        range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
        random_index = random.sample(range_list, crops)
        for j in range(crops):
            r = random_index[j]
            Gauss_data_Asperger.append(temp[r - fix_seq_length:r])
            # data_Asperger.append(temp[r - fix_seq_length:r])
            # data_Asperger.append(all_data[f][0][r - fix_seq_length:r])
            # agument_label_Asperger.append(label-1)
            # agument_label_Asperger.append(label-1)
    Gauss_data_Asperger = np.array(Gauss_data_Asperger)
    Gauss_Asperger_label=np.ones(Gauss_data_Asperger.shape[0])



    #PDD_NOS数据增强
    random.shuffle(flist_PDD_NOS)
    for f in flist_PDD_NOS:
        temp=all_data[f][0]
        noise = np.random.standard_normal(size=(all_data[f][0].shape[0], all_data[f][0].shape[1]))
        temp=temp+noise
        label=all_data[f][1]
        max_seq_length =temp.shape[0]
        range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
        random_index = random.sample(range_list, crops)
        for j in range(crops):
            r = random_index[j]
            # data_PDD_NOS.append(temp[r - fix_seq_length:r])
            data_PDD_NOS.append(all_data[f][0][r - fix_seq_length:r])

    data_PDD_NOS = np.array(data_PDD_NOS)
    print(data_PDD_NOS.shape)
    generator_model = Generator()
    discriminator_model = Discriminator()
    optimizer_G = optim.Adam(generator_model.parameters(), lr=0.0001)
    optimizer_D = optim.Adam(discriminator_model.parameters(), lr=0.0001)
    data_PDD_NOS=torch.from_numpy(data_PDD_NOS)
    data_PDD_NOS=data_PDD_NOS.to(torch.float32)
    reshape_data_PDD_NOS=data_PDD_NOS.view(data_PDD_NOS.size(0), -1)
    data_PDD_NOS=data_PDD_NOS.numpy()
    data_PDD_NOS_label=np.ones(data_PDD_NOS.shape[0])
    data_PDD_NOS_label += 1
    print(data_PDD_NOS_label)
    print(data_PDD_NOS_label.shape)
    agument_data_PDD_NOS = Gan_augment_train2(reshape_data_PDD_NOS, generator_model, discriminator_model,100, optimizer_G,optimizer_D,count=(count1-4*count3)*10)
    agument_data_PDD_NOS = torch.reshape(agument_data_PDD_NOS, (-1, 60, 200))
    print(agument_data_PDD_NOS.shape)
    agument_data_PDD_NOS=agument_data_PDD_NOS.detach().numpy()
    agument_PDD_NOS_label = np.ones(agument_data_PDD_NOS.shape[0])
    agument_PDD_NOS_label+=1
    print(agument_PDD_NOS_label.shape)



    for i in range(3):
        for f in flist_PDD_NOS:
            temp = all_data[f][0]
            noise = np.random.standard_normal(size=(all_data[f][0].shape[0], all_data[f][0].shape[1]))
            temp = temp + noise
            label = all_data[f][1]
            max_seq_length = temp.shape[0]
            range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
            random_index = random.sample(range_list, crops)
            for j in range(crops):
                r = random_index[j]
                # data_PDD_NOS.append(temp[r - fix_seq_length:r])
                Gauss_data_PDD_NOS.append(temp[r - fix_seq_length:r])
                # data_PDD_NOS.append(all_data[f][0][r - fix_seq_length:r])
    Gauss_data_PDD_NOS = np.array(Gauss_data_PDD_NOS)
    Gauss_PDD_NOS_label=np.ones(Gauss_data_PDD_NOS.shape[0])
    Gauss_PDD_NOS_label+=1


    flist_Autism=np.array(flist_Autism)
    random.shuffle(flist_Autism)
    for f in flist_Autism:
        max_seq_length = all_data[f][0].shape[0]
        range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
        random_index = random.sample(range_list, crops)
        for j in range(crops):
            r = random_index[j]
            augment_data_Autism.append(all_data[f][0][r - fix_seq_length:r])
            agument_label_Autism.append(all_data[f][1] - 1)
    augment_data.extend(agument_data_Asperger)
    augment_data.extend(data_Asperger)
    augment_data.extend(Gauss_data_Asperger)
    augment_data.extend(agument_data_PDD_NOS)
    augment_data.extend(data_PDD_NOS)
    augment_data.extend(Gauss_data_PDD_NOS)
    augment_data.extend(augment_data_Autism)
    agument_label.extend(agument_Asperger_label)
    agument_label.extend(data_Asperger_label)
    agument_label.extend(Gauss_Asperger_label)
    agument_label.extend(agument_PDD_NOS_label)
    agument_label.extend(data_PDD_NOS_label)
    agument_label.extend(Gauss_PDD_NOS_label)
    agument_label.extend(agument_label_Autism)


    print(np.array(augment_data).shape)
    print(np.array(agument_label).shape)

    return np.array(augment_data), np.array(agument_label)

#GGDB
def SubClass_data_agument6(all_data,sample_list,fix_seq_length,crops):
    augment_data = []
    agument_label = []
    flist_PDD_NOS=[]
    flist_Asperger=[]
    flist_Autism=[]

    data_Asperger=[]
    label_Asperger=[]
    data_PDD_NOS=[]
    label_PDD_NOS=[]

    data_Autism = []
    label_Autism =[]
    count1=0
    count2=0
    count3=0
    for f in sample_list:
        if (all_data[f][1] == 1):
            count1 += 1
            flist_Autism.append(f)
        elif (all_data[f][1] == 2):
            count2 += 1
            flist_Asperger.append(f)
        elif (all_data[f][1] == 3):
            count3 += 1
            flist_PDD_NOS.append(f)

    # Asperger 数据增强
    random.shuffle(flist_Asperger)
    for f in flist_Asperger:
        max_seq_length = all_data[f][0].shape[0]
        label = all_data[f][1]
        range_list = list(range(fix_seq_length + 1,int(max_seq_length)))
        random_index = random.sample(range_list,crops)
        for j in range(crops):
            r = random_index[j]
            data_Asperger.append(all_data[f][0][r - fix_seq_length:r])
            label_Asperger.append(label-1)
    data_Asperger=np.array(data_Asperger)
    print(data_Asperger.shape)
    generator_model=Generator()
    discriminator_model = Discriminator()
    optimizer_G = optim.Adam(generator_model.parameters(),lr=0.0001)
    optimizer_D = optim.Adam(discriminator_model.parameters(),lr=0.0001)
    data_Asperger=torch.from_numpy(data_Asperger)
    data_Asperger=data_Asperger.to(torch.float32)
    reshape_data_Asperger=data_Asperger.view(data_Asperger.size(0),-1)
    data_Asperger=data_Asperger.numpy()
    data_Asperger_label=np.ones(data_Asperger.shape[0])
    print(data_Asperger_label.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device="cpu"

    agument_data_Asperger=Gan_augment_train(reshape_data_Asperger,generator_model,discriminator_model,80,optimizer_G,optimizer_D,count=50,device=device)
    agument_data_Asperger=torch.reshape(agument_data_Asperger,(-1,90,200))
    print(agument_data_Asperger.shape)
    agument_data_Asperger=agument_data_Asperger.cpu().detach().numpy()
    # agument_data_Asperger=agument_data_Asperger.detach().numpy()
    agument_Asperger_label=np.ones(agument_data_Asperger.shape[0])
    print(agument_Asperger_label.shape)


    #PDD_NOS数据增强
    random.shuffle(flist_PDD_NOS)
    for f in flist_PDD_NOS:
        max_seq_length = all_data[f][0].shape[0]
        label = all_data[f][1]
        range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
        random_index = random.sample(range_list, crops)
        for j in range(crops):
            r = random_index[j]
            data_PDD_NOS.append(all_data[f][0][r - fix_seq_length:r])
            label_PDD_NOS.append(label-1)

    data_PDD_NOS = np.array(data_PDD_NOS)
    print(data_PDD_NOS.shape)
    generator_model = Generator()
    discriminator_model = Discriminator()
    optimizer_G = optim.Adam(generator_model.parameters(), lr=0.0001)
    optimizer_D = optim.Adam(discriminator_model.parameters(), lr=0.0001)
    data_PDD_NOS=torch.from_numpy(data_PDD_NOS)
    data_PDD_NOS=data_PDD_NOS.to(torch.float32)
    reshape_data_PDD_NOS=data_PDD_NOS.view(data_PDD_NOS.size(0), -1)
    data_PDD_NOS=data_PDD_NOS.numpy()
    data_PDD_NOS_label=np.ones(data_PDD_NOS.shape[0])
    data_PDD_NOS_label += 1
    # print(data_PDD_NOS_label)
    print(data_PDD_NOS_label.shape)
    agument_data_PDD_NOS = Gan_augment_train2(reshape_data_PDD_NOS, generator_model, discriminator_model,80,optimizer_G,optimizer_D,count=(count1-count3)*10)
    agument_data_PDD_NOS = torch.reshape(agument_data_PDD_NOS, (-1, 90, 200))
    print(agument_data_PDD_NOS.shape)
    agument_data_PDD_NOS=agument_data_PDD_NOS.detach().numpy()
    agument_PDD_NOS_label = np.ones(agument_data_PDD_NOS.shape[0])
    agument_PDD_NOS_label+=1
    print(agument_PDD_NOS_label.shape)


    flist_Autism=np.array(flist_Autism)
    random.shuffle(flist_Autism)
    for f in flist_Autism:
        max_seq_length = all_data[f][0].shape[0]
        range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
        random_index = random.sample(range_list, crops)
        for j in range(crops):
            r = random_index[j]
            data_Autism.append(all_data[f][0][r - fix_seq_length:r])
            label_Autism.append(all_data[f][1] - 1)
    augment_data.extend(agument_data_Asperger)
    augment_data.extend(data_Asperger)
    augment_data.extend(agument_data_PDD_NOS)
    augment_data.extend(data_PDD_NOS)
    augment_data.extend(data_Autism)
    agument_label.extend(agument_Asperger_label)
    agument_label.extend(data_Asperger_label)
    agument_label.extend(agument_PDD_NOS_label)
    agument_label.extend(data_PDD_NOS_label)
    agument_label.extend(label_Autism)

    print(np.array(data_Autism).shape)

    print(np.array(augment_data).shape)
    print(np.array(agument_label).shape)

    return np.array(augment_data), np.array(agument_label)



def drawPicture(all_data,sample_list,fix_seq_length,crops):
    augment_data = []
    agument_label = []
    flist_PDD_NOS=[]
    flist_Asperger=[]
    flist_Autism=[]

    data_Asperger=[]
    label_Asperger=[]
    data_PDD_NOS=[]
    label_PDD_NOS=[]

    data_Autism = []
    label_Autism =[]
    count1=0
    count2=0
    count3=0
    for f in sample_list:
        if (all_data[f][1] == 1):
            count1 += 1
            flist_Autism.append(f)
        elif (all_data[f][1] == 2):
            count2 += 1
            flist_Asperger.append(f)
        elif (all_data[f][1] == 3):
            count3 += 1
            flist_PDD_NOS.append(f)

    # Asperger 数据增强
    random.shuffle(flist_Asperger)
    for f in flist_Asperger:
        max_seq_length = all_data[f][0].shape[0]
        label = all_data[f][1]
        range_list = list(range(fix_seq_length + 1,int(max_seq_length)))
        random_index = random.sample(range_list,crops)
        for j in range(crops):
            r = random_index[j]
            data_Asperger.append(all_data[f][0][r - fix_seq_length:r])
            # label_Asperger.append(label-1)
            label_Asperger.append(label)
    data_Asperger=np.array(data_Asperger)
    print("data_Asperger.shape",data_Asperger.shape)
    generator_model=Generator()
    discriminator_model = Discriminator()
    optimizer_G = optim.Adam(generator_model.parameters(),lr=0.0001)
    optimizer_D = optim.Adam(discriminator_model.parameters(),lr=0.0001)
    data_Asperger=torch.from_numpy(data_Asperger)
    data_Asperger=data_Asperger.to(torch.float32)
    reshape_data_Asperger=data_Asperger.view(data_Asperger.size(0),-1)
    data_Asperger=data_Asperger.numpy()
    # data_Asperger_label=np.ones(data_Asperger.shape[0])
    # data_Asperger_label+=2
    # print(data_Asperger_label.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device="cpu"

    agument_data_Asperger=Gan_augment_train(reshape_data_Asperger,generator_model,discriminator_model,80,optimizer_G,optimizer_D,count=100,device=device)
    agument_data_Asperger=torch.reshape(agument_data_Asperger,(-1,90,200))
    print(agument_data_Asperger.shape)
    agument_data_Asperger=agument_data_Asperger.cpu().detach().numpy()
    # agument_data_Asperger=agument_data_Asperger.detach().numpy()
    agument_Asperger_label=np.ones(agument_data_Asperger.shape[0])
    agument_Asperger_label+=3
    print(agument_Asperger_label.shape)


    #PDD_NOS数据增强
    random.shuffle(flist_PDD_NOS)
    for f in flist_PDD_NOS:
        max_seq_length = all_data[f][0].shape[0]
        label = all_data[f][1]
        range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
        random_index = random.sample(range_list, crops)
        for j in range(crops):
            r = random_index[j]
            data_PDD_NOS.append(all_data[f][0][r - fix_seq_length:r])
            # label_PDD_NOS.append(label-1)
            label_PDD_NOS.append(label)

    data_PDD_NOS = np.array(data_PDD_NOS)
    print(data_PDD_NOS.shape)
    generator_model = Generator()
    discriminator_model = Discriminator()
    optimizer_G = optim.Adam(generator_model.parameters(), lr=0.0001)
    optimizer_D = optim.Adam(discriminator_model.parameters(), lr=0.0001)
    data_PDD_NOS=torch.from_numpy(data_PDD_NOS)
    data_PDD_NOS=data_PDD_NOS.to(torch.float32)
    reshape_data_PDD_NOS=data_PDD_NOS.view(data_PDD_NOS.size(0), -1)
    data_PDD_NOS=data_PDD_NOS.numpy()
    # data_PDD_NOS_label=np.ones(data_PDD_NOS.shape[0])
    # data_PDD_NOS_label += 1
    # print(data_PDD_NOS_label)
    # print(data_PDD_NOS_label.shape)
    agument_data_PDD_NOS = Gan_augment_train2(reshape_data_PDD_NOS, generator_model, discriminator_model,80,optimizer_G,optimizer_D,count=50)
    agument_data_PDD_NOS = torch.reshape(agument_data_PDD_NOS, (-1, 90, 200))
    print(agument_data_PDD_NOS.shape)
    agument_data_PDD_NOS=agument_data_PDD_NOS.detach().numpy()
    agument_PDD_NOS_label = np.ones(agument_data_PDD_NOS.shape[0])
    agument_PDD_NOS_label+=4
    print(agument_PDD_NOS_label.shape)


    augment_data.extend(agument_data_Asperger)
    augment_data.extend(data_Asperger)
    augment_data.extend(agument_data_PDD_NOS)
    augment_data.extend(data_PDD_NOS)

    agument_label.extend(agument_Asperger_label)
    agument_label.extend(label_Asperger)
    agument_label.extend(agument_PDD_NOS_label)
    agument_label.extend(label_PDD_NOS)


    print(np.array(augment_data).shape)
    print(np.array(agument_label).shape)

    return np.array(augment_data), np.array(agument_label)





#GAN
def SubClass_data_agument7(all_data,sample_list,fix_seq_length,crops):
    augment_data = []
    agument_label = []
    flist_PDD_NOS=[]
    flist_Asperger=[]
    flist_Autism=[]

    data_Asperger=[]
    label_Asperger=[]
    data_PDD_NOS=[]
    label_PDD_NOS=[]

    data_Autism = []
    label_Autism =[]
    count1=0
    count2=0
    count3=0
    for f in sample_list:
        if (all_data[f][1] == 1):
            count1 += 1
            flist_Autism.append(f)
        elif (all_data[f][1] == 2):
            count2 += 1
            flist_Asperger.append(f)
        elif (all_data[f][1] == 3):
            count3 += 1
            flist_PDD_NOS.append(f)

    # Asperger 数据增强
    random.shuffle(flist_Asperger)
    for f in flist_Asperger:
        max_seq_length = all_data[f][0].shape[0]
        label = all_data[f][1]
        range_list = list(range(fix_seq_length + 1,int(max_seq_length)))
        random_index = random.sample(range_list,crops)
        for j in range(crops):
            r = random_index[j]
            data_Asperger.append(all_data[f][0][r - fix_seq_length:r])
            label_Asperger.append(label-1)
    data_Asperger=np.array(data_Asperger)
    print(data_Asperger.shape)
    generator_model=Generator()
    discriminator_model = Discriminator()
    optimizer_G = optim.Adam(generator_model.parameters(),lr=0.0001)
    optimizer_D = optim.Adam(discriminator_model.parameters(),lr=0.0001)
    data_Asperger=torch.from_numpy(data_Asperger)
    data_Asperger=data_Asperger.to(torch.float32)
    reshape_data_Asperger=data_Asperger.view(data_Asperger.size(0),-1)
    data_Asperger=data_Asperger.numpy()
    data_Asperger_label=np.ones(data_Asperger.shape[0])
    print(data_Asperger_label.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device="cpu"

    agument_data_Asperger=Gan_augment_train(reshape_data_Asperger,generator_model,discriminator_model,80,optimizer_G,optimizer_D,count=(count1-count2)*10,device=device)
    agument_data_Asperger=torch.reshape(agument_data_Asperger,(-1,90,200))
    print(agument_data_Asperger.shape)
    agument_data_Asperger=agument_data_Asperger.cpu().detach().numpy()
    # agument_data_Asperger=agument_data_Asperger.detach().numpy()
    agument_Asperger_label=np.ones(agument_data_Asperger.shape[0])
    print(agument_Asperger_label.shape)


    #PDD_NOS数据增强
    random.shuffle(flist_PDD_NOS)
    for f in flist_PDD_NOS:
        max_seq_length = all_data[f][0].shape[0]
        label = all_data[f][1]
        range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
        random_index = random.sample(range_list, crops)
        for j in range(crops):
            r = random_index[j]
            data_PDD_NOS.append(all_data[f][0][r - fix_seq_length:r])
            label_PDD_NOS.append(label-1)

    data_PDD_NOS = np.array(data_PDD_NOS)
    print(data_PDD_NOS.shape)
    generator_model = Generator()
    discriminator_model = Discriminator()
    optimizer_G = optim.Adam(generator_model.parameters(), lr=0.0001)
    optimizer_D = optim.Adam(discriminator_model.parameters(), lr=0.0001)
    data_PDD_NOS=torch.from_numpy(data_PDD_NOS)
    data_PDD_NOS=data_PDD_NOS.to(torch.float32)
    reshape_data_PDD_NOS=data_PDD_NOS.view(data_PDD_NOS.size(0), -1)
    data_PDD_NOS=data_PDD_NOS.numpy()
    data_PDD_NOS_label=np.ones(data_PDD_NOS.shape[0])
    data_PDD_NOS_label += 1
    # print(data_PDD_NOS_label)
    print(data_PDD_NOS_label.shape)
    agument_data_PDD_NOS = Gan_augment_train2(reshape_data_PDD_NOS, generator_model, discriminator_model,80,optimizer_G,optimizer_D,count=(count1-count3)*10)
    agument_data_PDD_NOS = torch.reshape(agument_data_PDD_NOS, (-1, 90, 200))
    print(agument_data_PDD_NOS.shape)
    agument_data_PDD_NOS=agument_data_PDD_NOS.detach().numpy()
    agument_PDD_NOS_label = np.ones(agument_data_PDD_NOS.shape[0])
    agument_PDD_NOS_label+=1
    print(agument_PDD_NOS_label.shape)


    flist_Autism=np.array(flist_Autism)
    random.shuffle(flist_Autism)
    for f in flist_Autism:
        max_seq_length = all_data[f][0].shape[0]
        range_list = list(range(fix_seq_length + 1, int(max_seq_length)))
        random_index = random.sample(range_list, crops)
        for j in range(crops):
            r = random_index[j]
            data_Autism.append(all_data[f][0][r - fix_seq_length:r])
            label_Autism.append(all_data[f][1] - 1)
    augment_data.extend(agument_data_Asperger)
    augment_data.extend(data_Asperger)
    augment_data.extend(agument_data_PDD_NOS)
    augment_data.extend(data_PDD_NOS)
    augment_data.extend(data_Autism)
    agument_label.extend(agument_Asperger_label)
    agument_label.extend(data_Asperger_label)
    agument_label.extend(agument_PDD_NOS_label)
    agument_label.extend(data_PDD_NOS_label)
    agument_label.extend(label_Autism)

    print(np.array(data_Autism).shape)

    print(np.array(augment_data).shape)
    print(np.array(agument_label).shape)

    return np.array(augment_data), np.array(agument_label)




def feature_normalize(data):
    mu = np.mean(data)
    std = np.std(data)
    return (data - mu)/std

def get_data(all_data,sample_list):
    data=None
    label=None
    for f in sample_list:
        if data is None:
            data = np.expand_dims(feature_normalize(all_data[f][0][0:116]),axis=0)
            print(data.shape)
            label=all_data[f][1]
        else:
            temp=np.expand_dims(feature_normalize(all_data[f][0][0:116]),axis=0)
            data = np.vstack((data,temp))
            # print(data.shape)
            label=np.append(label,all_data[f][1])

    return data,label


#crops为切片的片数
#def get_data2(all_data,sample_list,crops):
def get_data2(all_data,sample_list):

    data=None
    label=None
    for f in sample_list:
        max_seq_length = all_data[f][0].shape[0]
        if data is None:
            #设置切片的片数
            for i in  range(5):
                j=np.random.randint(0,10)
                if data is None:
                    if (max_seq_length >= 100):
                        data = np.expand_dims(feature_normalize(all_data[f][0][j:j+68]),axis=0)
                        # print(data.shape)
                        label=all_data[f][1]
                else:
                    if (max_seq_length >= 100):
                        temp=np.expand_dims(feature_normalize(all_data[f][0][j:j+68]),axis=0)
                        data = np.vstack((data, temp))

                        label = np.append(label, all_data[f][1])
        else:
            for i in range(5):
                j = np.random.randint(0, 10)
                temp = np.expand_dims(feature_normalize(all_data[f][0][j:j + 68]), axis=0)
                data = np.vstack((data, temp))
                # print(data.shape)
                label = np.append(label, all_data[f][1])
    return data,label













def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# def get_data(all_data,sample_list):
#     data=None
#     label=None
#     for f in sample_list:
#         if data is None:
#             data = np.expand_dims(all_data[f][0][0:78],axis=0)
#             # print(data.shape)
#             label=all_data[f][1]
#         else:
#             temp=np.expand_dims(all_data[f][0][0:78],axis=0)
#             data = np.vstack((data,temp))
#             # print(data.shape)
#             label=np.append(label,all_data[f][1])
#     return data,label




def writeToExcel(file_path,new_list):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'test'
    for r in range(len(new_list)):
            for c in range(len(new_list[0])):
                ws.cell(r + 1, c + 1).value = new_list[r][c]
                # excel中的行和列是从1开始计数的，所以需要+1
    wb.save(file_path)  # 注意，写入后一定要保存
    print("成功写入文件: " + file_path + " !")
    return 1

