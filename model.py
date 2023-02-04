import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class time_series_Dataset(Dataset):
    def __init__(self, data=None, labels=None):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        data = self.data[index]
        label = (self.labels[index],)

        return torch.FloatTensor(data), torch.FloatTensor(label)

    def __len__(self):
        return len(self.labels)

class time_series_Dataset2(Dataset):
    def __init__(self, data=None,labels=None,Pheno_Info=None):
        self.data =data
        self.labels =labels
        self.Pheno_Info=Pheno_Info

    def __getitem__(self, index):
        data = self.data[index]
        label = (self.labels[index],)
        Pheno_Info=(self.Pheno_Info[index])


        return torch.FloatTensor(data), torch.FloatTensor(label),torch.FloatTensor(Pheno_Info)

    def __len__(self):
        return len(self.labels)

def get_loader(data=None,labels=None,bath_size=8,num_workers=0,mode=None):
    dataset=time_series_Dataset(data=data,labels=labels)
    if mode=='train':
        data_loader=DataLoader(
            dataset=dataset,
            batch_size=bath_size,
            shuffle=True,
            num_workers=num_workers
        )
    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=bath_size,
            shuffle=False,
            num_workers=num_workers
        )
    return data_loader

def get_loader2(data=None,labels=None,Pheno_Info=None,bath_size=8,num_workers=0,mode=None):
    dataset=time_series_Dataset2(data=data,labels=labels,Pheno_Info=Pheno_Info)
    if mode=='train':
        data_loader=DataLoader(
            dataset=dataset,
            batch_size=bath_size,
            shuffle=True,
            num_workers=num_workers
        )
    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=bath_size,
            shuffle=False,
            num_workers=num_workers
        )
    return data_loader










class CNN1d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(CNN1d, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.layer1=nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.MaxPool1d(self.kernel_size),

        )
        # self.layer2 = nn.Sequential(
        #     nn.Conv1d(in_channels=self.out_channels,
        #               out_channels=(int)(self.out_channels/2),
        #               kernel_size=self.kernel_size,
        #               stride=self.stride,
        #               padding=self.padding),
        #     nn.BatchNorm1d((int)(self.out_channels/2)),
        #     nn.ReLU(),
        #     nn.MaxPool1d(self.kernel_size),
        #
        #  )

        #self.fc=nn.Linear(2176,2)
        self.fc=nn.Linear(12672,68)
        self.fc=nn.Linear(4224,68)

    def forward(self,x):
        out=self.layer1(x)
        #out=self.layer2(out)
        print(out.shape)
        out=out.view(out.size(0),-1)
        # print(out.shape)
        out=self.fc(out)

        return out


def CNN1d_train(model,num_epochs,train_loader,optimizer,criterion,device):
    train_losses = []
    model.train()
    for epoch in range(1, num_epochs + 1):
        for i, (batch_x,batch_y) in enumerate(train_loader):
            batch_y = batch_y.long()
            batch_y = batch_y.flatten()
            data, label = batch_x.to(device), batch_y.to(device)
            
            data=data.permute(0,2,1)
            
            optimizer.zero_grad()
            logits = model(data)
            print("logits.shape",logits.shape)
            print("labels.shape",label.shape)
            #print(logits)
            loss = criterion(logits, label)
            train_losses.append([loss.item()])
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print("Train_Whole_CNN1d, Epoch: {}/{}, Iter: {}/{}, Loss: {:.8f}".format(
                    (epoch), num_epochs, (i + 1), len(train_loader), loss
                ))

    return train_losses


def CNN1d_test(model,criterion,test_loader,device):
    test_loss, n_test, correct = 0.0, 0, 0
    total, sensi, speci, tot0, tot1 = 0, 0, 0, 0, 0
    # 存放所有的预测
    all_predss = []
    y_true, y_pred = [], []
    with torch.no_grad():
        model.eval()
        for i, (batch_x,batch_y) in enumerate(test_loader,1):
            data = batch_x.to(device)
            y_arr=batch_y.flatten().numpy()
            #y_arr = np.array(batch_y, dtype=np.int32)
            data=data.permute(0,2,1)
            logits = model(data)
            _, predicted = torch.max(logits.data, 1)
            pred=predicted.cpu().numpy()
            total += batch_y.size(0)
            for index in  range(len(pred)):
                if pred[index]==y_arr[index]:
                    correct+=1
            # correct += (predicted.cpu().numpy()==y_arr.flatten()).sum()
            # # print(correct)
            for inum in range(len(pred)):
                if y_arr[inum] == 1:
                    tot0 += 1
                    if pred[inum] == 1:
                        sensi += 1
                else:
                    tot1 += 1
                    if pred[inum] == 0:
                        speci += 1
    return np.array([100 * correct / total, 100 * sensi / tot0, 100 * speci / tot1])







class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(RNN,self).__init__()

        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers

        self.rnn=nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2
        )
        #
        # self.out=nn.Sequential(
        #     nn.Linear(self.hidden_size*2,(int)(self.hidden_size*2/2)),
        #     nn.ReLU(),
        #     nn.Linear((int)(self.hidden_size*2/2),(int)(self.hidden_size*2/4)),
        #     nn.ReLU(),
        #     nn.Linear((int)(self.hidden_size*2 / 2),2),
        # )

        # self.out1=nn.Linear(self.hidden_size,(int)(self.hidden_size/2))
        # self.relu=nn.ReLU()
        # self.out2 = nn.Linear((int)(self.hidden_size / 2), 2)
        # self.out3 = nn.Linear((int)(self.hidden_size / 4),2)
        self.out=nn.Linear(self.hidden_size,2)
    def forward(self,x,device):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)

        r_out,(h_n,h_c)=self.rnn(x,(h_0,c_0))
        # out=self.out1(r_out[:,-1,:])
        # out=self.relu(out)
        # out = self.out2(out)
        # out = self.relu(out)
        # out = self.out3(out)
        out=self.out(r_out[:,-1,:])
        return out


def Rnn_train(model,num_epochs,train_loader,optimizer,criterion,device):
    train_losses = []
    model.train()
    for epoch in range(1, num_epochs + 1):
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x=batch_x.view(-1,48,200)
            # print(batch_x.shape)
            batch_y = batch_y.long()
            batch_y = batch_y.flatten()
            data, label = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(data,device)
            # print("logits.shape",logits.shape)
            # print("labels.shape",label.shape)
            # print(logits)
            loss = criterion(logits, label)
            train_losses.append([loss.item()])
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print("Train Whole, Epoch: {}/{}, Iter: {}/{}, Loss: {:.8f}".format(
                    (epoch), num_epochs, (i + 1), len(train_loader), loss
                ))

    return train_losses

def Rnn_test(model,criterion,test_loader,device):
    test_loss, n_test, correct = 0.0, 0, 0
    total, sensi, speci, tot0, tot1 = 0, 0, 0, 0, 0
    # 存放所有的预测
    all_predss = []
    y_true, y_pred = [], []
    with torch.no_grad():
        model.eval()
        for i, (batch_x,batch_y) in enumerate(test_loader,1):
            data = batch_x.to(device)
            y_arr=batch_y.flatten().numpy()
            #y_arr = np.array(batch_y, dtype=np.int32)
            logits = model(data,device)
            _, predicted = torch.max(logits.data, 1)
            pred=predicted.cpu().numpy()
            total += batch_y.size(0)
            for index in  range(len(pred)):
                if pred[index]==y_arr[index]:
                    correct+=1
            # correct += (predicted.cpu().numpy()==y_arr.flatten()).sum()
            # # print(correct)
            for inum in range(len(pred)):
                if y_arr[inum] == 1:
                    tot0 += 1
                    if pred[inum] == 1:
                        sensi += 1
                else:
                    tot1 += 1
                    if pred[inum] == 0:
                        speci += 1
    return np.array([100 * correct / total, 100 * sensi / tot0, 100 * speci / tot1])




def a_norm(Q,K):
    m=torch.matmul(Q,K.transpose(2,1).float())
    m/=torch.sqrt(torch.tensor(Q.shape[-1]).float())

    return torch.softmax(m,-1)


def attention(Q,K,V):
    a=a_norm(Q,K)

    return torch.matmul(a,V)

class Value(nn.Module):
    def __init__(self,dim_input,dim_val):
        super(Value,self).__init__()
        self.dim_val=dim_val

        self.fc1=nn.Linear(dim_input,dim_val,bias=False)
        # self.fc2=nn.Linear(5,dim_val)

    def forward(self,x):
        x=self.fc1(x)
        # x=self.fc2(x)

        return x


class Key(nn.Module):
    def __init__(self,dim_input,dim_attn):
        super(Key,self).__init__()
        self.dim_attn=dim_attn

        self.fc1=nn.Linear(dim_input,dim_attn,bias=False)
        # self.fc2=nn.Linear(5,dim_attn)

    def forward(self,x):
        x=self.fc1(x)
        # x=self.fc2(x)

        return x

class Query(nn.Module):
    def __init__(self,dim_input,dim_attn):
        super(Query,self).__init__()
        self.dim_attn=dim_attn

        self.fc1=nn.Linear(dim_input,dim_attn,bias=False)
        # self.fc2=nn.Linear(5,dim_attn)

    def forward(self,x):
        x=self.fc1(x)

        return x







class AttentionBlock(torch.nn.Module):
    def __init__(self,dim_val,dim_attn):
        super(AttentionBlock,self).__init__()
        self.value=Value(dim_val,dim_val)
        self.Key=Key(dim_val,dim_attn)
        self.query=Query(dim_val,dim_attn)


    def forward(self,x,kv=None):
        if kv is None:
            return attention(self.query(x),self.Key(x),self.value(x))


        return attention(self.query(x),self.Key(kv),self.value(kv))



class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,dim_val,dim_attn,n_heads):
        super(MultiHeadAttentionBlock,self).__init__()
        self.heads=[]
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val,dim_attn))

        self.heads=nn.ModuleList(self.heads)

        self.fc=nn.Linear(n_heads*dim_val,dim_val,bias=False)


    def forward(self,x,kv=None):
        a=[]

        for h in  self.heads:
            a.append(h(x,kv=kv))

        a=torch.stack(a,dim=-1)
        a=a.flatten(start_dim=2)

        x=self.fc(a)

        return  x


class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super(PositionalEncoding,self).__init__()

        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(-1)

        div_term=torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


def getdata(batch_size, input_sequence_length, output_sequence_length):
    i = input_sequence_length + output_sequence_length

    t = torch.zeros(batch_size, 1).uniform_(0, 20 - i).int()
    b = torch.arange(-10, -10 + i).unsqueeze(0).repeat(batch_size, 1) + t

    s = torch.sigmoid(b.float())
    return s[:, :input_sequence_length].unsqueeze(-1), s[:, -output_sequence_length:]



class EncoderLayer(nn.Module):
    def __init__(self,dim_val,dim_attn,n_heads=1):
        super(EncoderLayer,self).__init__()
        self.attn=MultiHeadAttentionBlock(dim_val,dim_attn,n_heads)

        self.fc1=nn.Linear(dim_val,dim_val)
        self.fc2=nn.Linear(dim_val,dim_val)

        self.norm1=nn.LayerNorm(dim_val)
        self.norm2=nn.LayerNorm(dim_val)

    def forward(self,x):
        a=self.attn(x)
        x=self.norm1(x+a)

        a=self.fc1(F.elu(self.fc2(x)))
        x=self.norm2(x+a)

        return x


class DecoderLayer(nn.Module):
    def __init__(self,dim_val,dim_attn,n_heads=1):
        super(DecoderLayer,self).__init__()
        self.attn1=MultiHeadAttentionBlock(dim_val,dim_attn,n_heads)
        self.attn2=MultiHeadAttentionBlock(dim_val,dim_attn,n_heads)
        self.fc1=nn.Linear(dim_val,dim_val)
        self.fc2=nn.Linear(dim_val,dim_val)


        self.norm1=nn.LayerNorm(dim_val)
        self.norm2=nn.LayerNorm(dim_val)
        self.norm3=nn.LayerNorm(dim_val)

    def forward(self,x,enc):
        a=self.attn1(x)
        x=self.norm1(a+x)

        a = self.attn2(x, kv=enc)
        x = self.norm2(a + x)

        a = self.fc1(F.elu(self.fc2(x)))

        x = self.norm3(x + a)

        return x


class Transformer(nn.Module):
    def __init__(self,dim_val,dim_attn,input_size,dec_seq_len,out_seq_len,n_decoder_layers=1,n_encoder_layers=1,n_heads=1):
        super(Transformer,self).__init__()
        self.dec_seq_len=dec_seq_len


        self.encs=nn.ModuleList()
        for i in  range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val,dim_attn,n_heads))

        self.decs=nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val,dim_attn,n_heads))

        self.pos=PositionalEncoding(dim_val)

        self.enc_input_fc=nn.Linear(input_size,dim_val)
        self.dec_input_fc=nn.Linear(input_size,dim_val)

        self.out_fc=nn.Linear(dec_seq_len*dim_val,out_seq_len)

    def forward(self,x):
        #encoder
        e=self.encs[0](self.pos(self.enc_input_fc(x)))
        for enc in self.encs[1:]:
            e=enc(e)

        #decoder
        d=self.decs[0](self.dec_input_fc(x[:,-self.dec_seq_len:]),e)
        for dec in self.decs[1:]:
            d=dec(d,e)


        x=self.out_fc(d.flatten(start_dim=1))

        return  x



# def transformer_train()



