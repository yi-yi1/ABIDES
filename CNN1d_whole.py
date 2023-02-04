from Utils import *
import pyprind
import pickle
import warnings
from sklearn.model_selection import StratifiedKFold
from model import *
import torch.optim as optim
from datetime import datetime

warnings.filterwarnings('ignore')




#数据集的文件路径
data_path='E:/桌面/rois_cc200'
#将data_path路径下的所有文件名全部存放在flist列表中
flist=os.listdir(data_path)
print('flist.length',len(flist))

#修改flist列表中的文件名
for f in range(len(flist)):
    flist[f]=get_key(flist[f])

#获取表型数据
df_labels=pd.read_csv('./Phenotypic_V1_0b_preprocessed1.csv')
# 修改映射
# 1=Autism;  2=Control
# 1-->1   2-->0
df_labels.DX_GROUP = df_labels.DX_GROUP.map({1: 1, 2: 0})
print('labels.length',len(df_labels))

#用labels{}存放每个文件对应的标签
labels = {}
for row in df_labels.iterrows():
    # 获取每一行的文件ID 和标签
    file_id = row[1]['FILE_ID']
    y_label = row[1]['DX_GROUP']
    if file_id == 'no_filename':
        continue
    assert (file_id not in labels)
    labels[file_id] = y_label





if not os.path.exists('./cc200_data.pkl'):
    pbar = pyprind.ProgBar(len(flist))
    all_data={}
    for f in flist:
        label=get_label(f,labels)
        all_data[f]=(get_time_series_data(f,data_path),label)
        pbar.update()
    print('data_load finished')
    pickle.dump(all_data, open('./cc200_data.pkl', 'wb'))
    print('Saving to file finished')
else:
    all_data=pickle.load(open('./cc200_data.pkl', 'rb'))

# np.random.shuffle(flist)
# print(flist)

site=['Caltech','CMU','NYU','SDSU','Stanford','Trinity','UM','USM','Yale']
count=0
for i  in range(len(site)):
    for f in flist:
        # print(f)
        # print(site[i])
        if f.startswith(site[i]):
            count=count+1
            #print(count)
            # print(site[i])
            # print(np.asarray(all_data[f][0]).shape)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

p_fold=10
num_epochs=10
bath_size=8
result=[]
three_result=[]
sum_result=[]

start=datetime.now()
for i in  range(10):
    result = []
    three_result = []
    result.append(num_epochs)
    print("num_epochs",num_epochs)
    y_arr = np.array([get_label(f,labels) for f in flist])
    flist = np.array(flist)
    kk = 0
    kf = StratifiedKFold(n_splits=p_fold, random_state=10, shuffle=True)
    for kk, (train_index, test_index) in enumerate(kf.split(flist,y_arr)):
        train_data_flist,test_data_flist=flist[train_index],flist[test_index]

        train_data,train_label=get_data2(all_data=all_data, sample_list=train_data_flist)
        test_data,test_label=get_data2(all_data=all_data, sample_list=test_data_flist)
        # 将训练集打乱
        # np.random.seed(116)
        # np.random.shuffle(train_data)
        # np.random.seed(116)
        # np.random.shuffle(train_label)
        # test_data, test_label =data[test_index],label[test_index]
        train_loader = get_loader(data=train_data, labels=train_label,bath_size=bath_size,mode='train')
        test_loader = get_loader(data=test_data, labels=test_label,bath_size=bath_size,mode='test')
        #model=CNN1d(200,128,2,1,0)
        model=CNN1d(200,128,2,1,0)
        model.to(device)
        #明天再来更换一下损失函数
        # criterion=nn.BCELoss()
        criterion = nn.CrossEntropyLoss()

        optimizer =optim.Adam(model.parameters(), lr=0.001)
        train_loss = CNN1d_train(model, num_epochs, train_loader, optimizer, criterion, device)
        res_mlp = CNN1d_test(model, criterion, test_loader, device)
        print(res_mlp)
        result.append(res_mlp[0])
        three_result.append(res_mlp)

    print("averages:")
    result_mean = np.mean(np.array(result[1:]), axis=0)
    print(result_mean)
    print(np.mean(np.array(three_result), axis=0))
    result.append(result_mean)
    sum_result.append(result)
    print(sum_result)
    finish = datetime.now()
    print("用时(小时)：", float(((finish - start).seconds) / (60 * 60)))
    num_epochs += 10

writeToExcel('CNN1d_whole_5_slicing.xls',sum_result)