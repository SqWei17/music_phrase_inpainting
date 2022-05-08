import json
import random
import torch
import os
import numpy as np
import time
from torch import nn
from struacture import Musicmodel
# from data_loader import MusicArrayLoader
from torch.autograd import Variable
from torch.utils import data
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pretty_midi
import csv
from sklearn.model_selection import train_test_split
from data_loader import MusicArrayLoader
from sklearn.model_selection import train_test_split
import codecs
from sklearn.utils import shuffle

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'


def seed_torch(seed=1234):
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.cuda.manual_seed_all(seed)




class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs





# some initialization
with open('/hierarchy/model_config.json') as f:
    args = json.load(f)
if not os.path.isdir('log'):
    os.mkdir('log')


load_path = '/home/Deep-Music-Analogy-Demos/code/params1bar/val/best-valid-model.pt'
load_path_ = '/home/Deep-Music-Analogy-Demos/CODE/param8128kl/val/best-valid-model.pt'
load_path_2 = '/home/Deep-Music-Analogy-Demos/code/params1bar/val/best-valid-model.pt'
load_path_h = '/hierarchy/try/3param1283r8/val/best-valid-modelnext.pt'
load_path_4 = '/home/Deep-Music-Analogy-Demos/CODE/param4bar128dim0003fen/val/best-valid-model.pt'
# writer = SummaryWriter('log/{}'.format(args['modelname']))

print('Project initialized.', flush=True)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

model = Musicmodel(130, args['hidden_dim'], 3, args['pitch_dim'],
                   args['rhythm_dim'], 128, 128)
# musicvae = Musicmodel(130, args['hidden_dim'], 3, 12, args['pitch_dim'],args['rhythm_dim'], 32,128)


if args['if_parallel']:
    model = torch.nn.DataParallel(model, device_ids=[0,1,2])
optimizer1 = optim.Adam(model.parameters(), lr=args['lr'])
# optimizer2 = optim.Adam(mlp.parameters(), lr=args['lr'])
if args['decay'] > 0:
    scheduler = MinExponentialLR(optimizer1, gamma=args['decay'], minimum=1e-5)
if torch.cuda.is_available():
    print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print('CPU mode')



if os.path.exists(load_path_):
    checkpoint = torch.load(load_path_)
    model_dict = model.state_dict()
    for name in list(checkpoint['model'].keys()):
        print(name)
        checkpoint['model'][name.replace('module.musicvae8.', 'module.int_model_8.')] = checkpoint['model'].pop(name)
    state_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict.keys()}
    for k, v in checkpoint['model'].items():
        for name, p in model.named_parameters():
            if k == name:
                print(k)
                p.requires_grad = False
    print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    for p in model.parameters():
        print(p.requires_grad)
    print('load成功')
else:
    step, pre_epoch = 0, 0
    print('无保存模型，将从头开始训练！')

if os.path.exists(load_path_h):
    checkpoint = torch.load(load_path_h)
    model_dict = model.state_dict()
    for name in list(checkpoint['model'].keys()):
        print(name)
        checkpoint['model'][name.replace('module.Musicmodel.', 'module.Musicmodel_h.')] = checkpoint['model'].pop(name)
        #checkpoint['model'][name.replace('module.Musicmodel_h.decodep84.', 'module.decodep84.')] = checkpoint['model'].pop(name)
        #checkpoint['model'][name.replace('module.Musicmodel_h.decoder84.', 'module.decoder84.')] = checkpoint['model'].pop(name)
    state_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict.keys()}
    for k, v in checkpoint['model'].items():
        for name, p in model.named_parameters():
            if k == name:
                print(k)
                p.requires_grad = False
    print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    for p in model.parameters():
        print(p.requires_grad)
    print('load成功')
else:
    step, pre_epoch = 0, 0
    print('无保存模型，将从头开始训练！')



if os.path.exists(load_path):
    checkpoint = torch.load(load_path)
    model_dict = model.state_dict()
    for name in list(checkpoint['model'].keys()):
        print(name)
        checkpoint['model'][name.replace('module.', 'module.int_model_1.')] = checkpoint['model'].pop(name)
    state_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict.keys()}
    for k, v in checkpoint['model'].items():
        for name, p in model.named_parameters():
            if k == name:
                print(k)
                p.requires_grad = False
    print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    for p in model.parameters():
        print(p.requires_grad)
    print('load成功')
else:
    step, pre_epoch = 0, 0
    print('无保存模型，将从头开始训练！')

if os.path.exists(load_path_4):
    checkpoint = torch.load(load_path_4)
    model_dict = model.state_dict()
    for name in list(checkpoint['model'].keys()):
        print(name)
        checkpoint['model'][name.replace('module.musicvae4.', 'module.int_model_4.')] = checkpoint['model'].pop(
            name)
    state_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict.keys()}
    for k, v in checkpoint['model'].items():
        for name, p in model.named_parameters():
            if k == name:
                print(k)
                p.requires_grad = False
    print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    for p in model.parameters():
        print(p.requires_grad)
else:
    step, pre_epoch = 0, 0
    print('无保存模型，将从头开始训练！')







dataset = np.load("/boot/Deep-Music-Analogy-Demos/code/data.npy", allow_pickle=True)
melody = dataset[0]
chord = dataset[1]
#melody,chord = shuffle(melody,chord)
n_number = int(0.95* melody.shape[0])



def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")


def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N

step_train = 0
step_val = 0
pre_epoch = 0
train_data = Dataset(melody[:n_number], chord[:n_number])
val_data = Dataset_val(melody[n_number:], chord[n_number:])
train_loader = data.DataLoader(train_data, batch_size=384, shuffle=True)
val_loader = data.DataLoader(val_data, batch_size=384, shuffle=True)
def loss_function(recon9, recon_rhythm9,
                  target_tensor9, rhythm_target9,recon10, recon_rhythm10,
                  target_tensor10, rhythm_target10,recon11, recon_rhythm11,
                  target_tensor11, rhythm_target11,recon12, recon_rhythm12,
                  target_tensor12, rhythm_target12,distribution_01,distribution_11,cpcpm1,cpcrm1,cpcpm2,cpcrm2,cpcpm3,cpcrm3,cpcpm4,cpcrm4,beta=.1):

    CE91 = F.nll_loss(
        recon9.view(-1, recon9.size(-1)),
        target_tensor9,
        reduction='elementwise_mean')
    CE92 = F.nll_loss(
        recon_rhythm9.view(-1, recon_rhythm9.size(-1)),
        rhythm_target9,
        reduction='elementwise_mean')
    CE101 = F.nll_loss(
        recon10.view(-1, recon10.size(-1)),
        target_tensor10,
        reduction='elementwise_mean')
    CE102 = F.nll_loss(
        recon_rhythm10.view(-1, recon_rhythm10.size(-1)),
        rhythm_target10,
        reduction='elementwise_mean')
    CE111 = F.nll_loss(
        recon11.view(-1, recon11.size(-1)),
        target_tensor11,
        reduction='elementwise_mean')
    CE112 = F.nll_loss(
        recon_rhythm11.view(-1, recon_rhythm11.size(-1)),
        rhythm_target11,
        reduction='elementwise_mean')
    CE121 = F.nll_loss(
        recon12.view(-1, recon12.size(-1)),
        target_tensor12,
        reduction='elementwise_mean')
    CE122 = F.nll_loss(
        recon_rhythm12.view(-1, recon_rhythm12.size(-1)),
        rhythm_target12,
        reduction='elementwise_mean')

    normal1 = std_normal(distribution_01.mean.size())
    normal2 = std_normal(distribution_11.mean.size())
    KLD1 = kl_divergence(distribution_01, normal1).mean()
    KLD2 = kl_divergence(distribution_11, normal2).mean()



    max_indices = recon9.view(-1, recon9.size(-1)).max(-1)[-1]
    #     print(max_indices)
    correct = max_indices == target_tensor9
    acc9 = torch.sum(correct.float()) / target_tensor9.size(0)

    max_indices2 = recon_rhythm9.view(-1, recon_rhythm9.size(-1)).max(-1)[-1]
    #     print(max_indices)
    correct2 = max_indices2 == rhythm_target9
    racc9 = torch.sum(correct2.float()) / rhythm_target9.size(0)

    max_indices = recon10.view(-1, recon10.size(-1)).max(-1)[-1]
    #     print(max_indices)
    correct = max_indices == target_tensor10
    acc10 = torch.sum(correct.float()) / target_tensor10.size(0)

    max_indices2 = recon_rhythm10.view(-1, recon_rhythm10.size(-1)).max(-1)[-1]
    #     print(max_indices)
    correct2 = max_indices2 == rhythm_target10
    racc10 = torch.sum(correct2.float()) / rhythm_target10.size(0)

    max_indices = recon11.view(-1, recon11.size(-1)).max(-1)[-1]
    #     print(max_indices)
    correct = max_indices == target_tensor11
    acc11 = torch.sum(correct.float()) / target_tensor11.size(0)

    max_indices2 = recon_rhythm11.view(-1, recon_rhythm11.size(-1)).max(-1)[-1]
    #     print(max_indices)
    correct2 = max_indices2 == rhythm_target11
    racc11 = torch.sum(correct2.float()) / rhythm_target11.size(0)

    max_indices = recon12.view(-1, recon12.size(-1)).max(-1)[-1]
    #     print(max_indices)
    correct = max_indices == target_tensor12
    acc12 = torch.sum(correct.float()) / target_tensor12.size(0)

    max_indices2 = recon_rhythm12.view(-1, recon_rhythm12.size(-1)).max(-1)[-1]
    #     print(max_indices)
    correct2 = max_indices2 == rhythm_target12
    racc12 = torch.sum(correct2.float()) / rhythm_target12.size(0)



    cpc91 = cpcpm1.mean()
    cpc92 = cpcrm1.mean()
    cpc101 = cpcpm2.mean()
    cpc102 = cpcrm2.mean()
    cpc111 = cpcpm3.mean()
    cpc112 = cpcrm3.mean()
    cpc121 = cpcpm4.mean()
    cpc122 = cpcrm4.mean()


    #loss = CE1 + CE2+CE3 + CE4+CE5 + CE6+CE7 + CE8+p21+r21#+d1+d2+d3+d4+d5+d1_+d2_+d3_+d4_+d5_#+beta*(KLD1+KLD2)
    loss =  CE91 + CE92+CE101 + CE102+CE111 + CE112+CE121 + CE122+cpc91+cpc92+cpc101+cpc102+cpc111+cpc112+cpc121+cpc122#/home/sqwei/inpainting_lstm/train_1.py+0.1*(KLD1+KLD2)#+dp5+dr5+dp4+dr4

    return loss, CE91,CE92,  acc9, racc9,acc10, racc10,acc11, racc11,acc12, racc12


def train(train_loader,model, optimizer1, epoch,step_train):
    # batch, c = dl.get_batch(args['batch_size'])
    # batch = batch.numpy()
    # c = c.numpy()
    batch_loss = []
    accc = []
    raccc = []
    accc4 = []
    raccc4 = []
    accc3 = []
    raccc3 = []
    accc2 = []
    raccc2 = []
    ce1 = []
    ce2 = []
    a = 1
    model.train()
    epoch_loss = 0.


    for mel1, ch1 in train_loader:
        print(epoch)
        mel1 = mel1.numpy()
        ch1 = ch1.numpy()
        encode_tensor = torch.from_numpy(mel1).float()
        c = torch.from_numpy(ch1[:, :, :]).float()
        # et = encode_tensor.contiguous()
        before_piece = encode_tensor[:,0:128,:].contiguous()
        before_piece_ = encode_tensor[:,192:256,:].contiguous()
        before_piece2_ = encode_tensor[:, 0:32, :].contiguous()
        before_piece1_ = encode_tensor[:, 32:64, :].contiguous()
        before_piece1 = encode_tensor[:, 64:96, :].contiguous()
        before_piece2 = encode_tensor[:, 96:128, :].contiguous()
        after_piece2_ = encode_tensor[:, 192:224, :].contiguous()
        after_piece1_ = encode_tensor[:, 224:256, :].contiguous()
        #after_piece1 = encode_tensor[:, 224:256, :].contiguous()
        #after_piece2 = encode_tensor[:, 256:288, :].contiguous()
        c_before = c[:,0:128,:].contiguous()
        c_before2_ = c[:, 0:32, :].contiguous()
        c_before1_ = c[:, 32:64, :].contiguous()
        c_before1 = c[:, 64:96, :].contiguous()
        c_before2 = c[:, 96:128, :].contiguous()
        c_after = c[:,192:256,:].contiguous()
        c_middle = c[:,128:192,:].contiguous()


        c_after2_ = c[:, 192:224, :].contiguous()
        c_after1_ = c[:, 224:256, :].contiguous()
        #c_after1 = c[:, 224:256, :].contiguous()
        #c_after2 = c[:, 256:288, :].contiguous()

        #after_piece = encode_tensor[:,160:288,:].contiguous()

        et1 = encode_tensor[:,128:192,:].contiguous()
        et9 = encode_tensor[:, 128:144, :].contiguous()
        et10 = encode_tensor[:, 144:160, :].contiguous()
        et11 = encode_tensor[:, 160:176, :].contiguous()
        et12 = encode_tensor[:, 176:192, :].contiguous()
        target_tensor5 = et1.view(-1, et1.size(-1)).max(-1)[1]
        target_tensor9 = et9.view(-1, et9.size(-1)).max(-1)[1]
        target_tensor10 = et10.view(-1, et10.size(-1)).max(-1)[1]
        target_tensor11 = et11.view(-1, et11.size(-1)).max(-1)[1]
        target_tensor12 = et12.view(-1, et12.size(-1)).max(-1)[1]

        target_tensor2 = before_piece1_.view(-1, before_piece1_.size(-1)).max(-1)[1]
        target_tensor3 = before_piece1.view(-1, before_piece1.size(-1)).max(-1)[1]
        target_tensor4 = before_piece2.view(-1, before_piece2.size(-1)).max(-1)[1]


        rhythm_target9 = np.expand_dims(et9[:, :, :-2].sum(-1), -1)
        rhythm_target9 = np.concatenate((rhythm_target9, et9[:, :, -2:]), -1)
        rhythm_target9 = torch.from_numpy(rhythm_target9).float()
        rhythm_target9 = rhythm_target9.view(-1, rhythm_target9.size(-1)).max(-1)[1]

        rhythm_target10 = np.expand_dims(et10[:, :, :-2].sum(-1), -1)
        rhythm_target10 = np.concatenate((rhythm_target10, et10[:, :, -2:]), -1)
        rhythm_target10 = torch.from_numpy(rhythm_target10).float()
        rhythm_target10 = rhythm_target10.view(-1, rhythm_target10.size(-1)).max(-1)[1]

        rhythm_target11 = np.expand_dims(et11[:, :, :-2].sum(-1), -1)
        rhythm_target11 = np.concatenate((rhythm_target11, et11[:, :, -2:]), -1)
        rhythm_target11 = torch.from_numpy(rhythm_target11).float()
        rhythm_target11 = rhythm_target11.view(-1, rhythm_target11.size(-1)).max(-1)[1]

        rhythm_target12 = np.expand_dims(et12[:, :, :-2].sum(-1), -1)
        rhythm_target12 = np.concatenate((rhythm_target12, et12[:, :, -2:]), -1)
        rhythm_target12 = torch.from_numpy(rhythm_target12).float()
        rhythm_target12 = rhythm_target12.view(-1, rhythm_target12.size(-1)).max(-1)[1]

        if torch.cuda.is_available():
            before_piece = before_piece.cuda()
            #after_piece = after_piece.cuda()
            encode_tensor = encode_tensor.cuda()
            target_tensor2 = target_tensor2.cuda()
            target_tensor3 = target_tensor3.cuda()
            target_tensor4 = target_tensor4.cuda()
            target_tensor5 = target_tensor5.cuda()
            target_tensor9 = target_tensor9.cuda()
            target_tensor10 = target_tensor10.cuda()
            target_tensor11 = target_tensor11.cuda()
            target_tensor12 = target_tensor12.cuda()
            #c = c.cuda()

            rhythm_target9 = rhythm_target9.cuda()
            rhythm_target10 = rhythm_target10.cuda()
            rhythm_target11 = rhythm_target11.cuda()
            rhythm_target12 = rhythm_target12.cuda()
        optimizer1.zero_grad()
        recon9, recon_rhythm9,recon10, recon_rhythm10,recon11, recon_rhythm11,recon12, recon_rhythm12,dis1m,dis1a,dis2m,dis2s,cpcpm1,cpcrm1,cpcpm2,cpcrm2,cpcpm3,cpcrm3,cpcpm4,cpcrm4 = model(before_piece2_,before_piece1_,before_piece1,before_piece2,before_piece,before_piece_,et1,c_before2_,c_before1_,c_before1,c_before2,c_before,c_after,c_middle,after_piece2_,after_piece1_,c_after2_,c_after1_,epoch)
        dis1 = Normal(dis1m, dis1a)
        dis2 = Normal(dis2m, dis2s)


        # valid = Variable(FloatTensor(recon0.size(0), 1).fill_(1.0), requires_grad=False)
        # fake = Variable(FloatTensor(recon0.size(0), 1).fill_(0.0), requires_grad=False)

        sum_loss, CE1,CE2, acc, racc,acc2, racc2 ,acc3,racc3,acc4,racc4= loss_function(recon9, recon_rhythm9,target_tensor9,rhythm_target9,recon10, recon_rhythm10,target_tensor10,rhythm_target10,recon11, recon_rhythm11,target_tensor11,rhythm_target11,recon12, recon_rhythm12,target_tensor12,rhythm_target12,dis1,dis2,cpcpm1,cpcrm1,cpcpm2,cpcrm2,cpcpm3,cpcrm3,cpcpm4,cpcrm4,beta=0.1)

        sum_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer1.step()

        batch_loss.append(sum_loss.item())
        ce1.append(CE1.item())
        ce2.append(CE2.item())
        accc.append(acc.item())
        raccc.append(racc.item())
        accc2.append(acc2.item())
        raccc2.append(racc2.item())
        accc3.append(acc3.item())
        raccc3.append(racc3.item())
        accc4.append(acc4.item())
        raccc4.append(racc4.item())

        if args['decay'] > 0:
            scheduler.step()
        # dl.shuffle_samples()
        array1 = np.array(batch_loss)
        array2 = np.array(ce1)
        array3 = np.array(ce2)
        array4 = np.array(accc)
        array5 = np.array(raccc)
        array6 = np.array(accc2)
        array7 = np.array(raccc2)
        array8 = np.array(accc3)
        array9 = np.array(raccc3)
        array10 = np.array(accc4)
        array11 = np.array(raccc4)
        step_train += 1

    return (array1.mean(), array2.mean(), array3.mean(), array4.mean(), array5.mean(), array6.mean(), array7.mean(), array8.mean(), array9.mean(), array10.mean(), array11.mean(),step_train)



best_valid_loss = float('inf')
start = time.time()
while epoch <100:
    seed_torch(77)



    print(epoch)

    train_loss, CE1,CE2, acc, racc, acc2, racc2, acc3, racc3, acc4, racc4,step_train = train(train_loader,model,optimizer1, epoch,step_train)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    pre_epoch = epoch
    print(pre_epoch)
    state1 = {'model': model.cpu().state_dict(), 'optimizer': optimizer1.state_dict(), 'epoch': epoch,
              'scheduler_state_dict': scheduler.state_dict()}



    if epoch % 1 == 0:
        save_path1 = '/buquan/bu4_84/param_aug1295/train/model8.pt'
        # save_path1 = 'param/vaecpccos22/{}.pt'.format(epoch)
        torch.save(state1, save_path1)
        print("Model saved")

    if torch.cuda.is_available():
        model.cuda()



    epoch = epoch + 1
