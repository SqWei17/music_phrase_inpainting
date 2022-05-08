import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
import model as models

class Musicmodel(nn.Module):
    def __init__(self,
                 roll_dims,
                 hidden_dims,
                 condition_dims,
                 z1_dims,
                 z2_dims,
                 n_step,
                 p_queuesize,
                 k=1000):
        super(Musicmodel, self).__init__()

        self.queue_size_total = p_queuesize
        #self.loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        self.loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)


        self.Wp21 = nn.Parameter(torch.randn(128, 128))
        self.Wr21 = nn.Parameter(torch.randn(128, 128))
        self.Wp214 = nn.Parameter(torch.randn(128, 128))
        self.Wr214 = nn.Parameter(torch.randn(128, 128))

        # assert self.queue_size % args.batch_size == 0  # for simplicity
        self.int_model_8 = models.VAEbar8(130, hidden_dims, 3, 12, 128, 128,128)
        self.int_model_8.requires_grad = False
        self.int_model_4 = models.VAEbar4(130, hidden_dims, 3, 12, 128, 128,64)
        self.int_model_4.requires_grad = False
        # self.int_model_4.encoder.parameters().requires_grad = False
        self.int_model_1 = models.VAEbar1(130, hidden_dims, 3, 12, 128, 128, 16)
        self.int_model_1.requires_grad = False
        self.int_model_1_ = models.VAEbar1_(130, hidden_dims, 3, 12, 128, 128, 16)
        self.preddlr = models.Interpolationlr(1024, 128, 128, 8,3)
        self.preddrl = models.Interpolationrl(1024, 128, 128, 4,3)

        self.mid_linearp = nn.Linear(256,128)
        self.mid_linearr = nn.Linear(256,128)
        self.register_buffer("p_queue", torch.randn(self.queue_size_total, 128))

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("p_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("r_queue", torch.randn(self.queue_size_total, 128))

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("r_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_pqueue(self, feat_p):
        '''
        Update the memory / queue.
        Add batch to end of most recent sample index and remove the oldest samples in the queue.
        Store location of most recent sample index (ptr).
        Taken from: https://github.com/facebookresearch/moco
        args:
            feat_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.
        '''

        batch_size = feat_p.size(0)

        ptr = int(self.p_queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        self.p_queue[ptr:ptr + batch_size, :] = feat_p

        # move pointer along to end of current batch
        ptr = (ptr + batch_size) % self.queue_size_total

        # Store queue pointer as register_buffer
        self.p_queue_ptr[0] = ptr

    @torch.no_grad()
    def update_rqueue(self, feat_p):
        '''
        Update the memory / queue.
        Add batch to end of most recent sample index and remove the oldest samples in the queue.
        Store location of most recent sample index (ptr).
        Taken from: https://github.com/facebookresearch/moco
        args:
            feat_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.
        '''

        batch_size = feat_p.size(0)

        ptr = int(self.r_queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        self.r_queue[ptr:ptr + batch_size, :] = feat_p

        # move pointer along to end of current batch
        ptr = (ptr + batch_size) % self.queue_size_total

        # Store queue pointer as register_buffer
        self.r_queue_ptr[0] = ptr






    def InfoNCE_logitsp21no(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
      #  pos1 = torch.mm(f_a, self.Wp21)
        pos1 = torch.mm(f_a, f_1.transpose(1, 0)) / tau
       # neg1 = torch.mm(f_a, self.Wp21)
        neg1 = torch.mm(f_a, f_2.transpose(1, 0)) / tau
        #neg2 = torch.mm(f_a, self.Wp21)
        neg2 = torch.mm(f_a, f_3.transpose(1, 0)) / tau
       # neg3 = torch.mm(f_a, self.Wp21)
        neg3 = torch.mm(f_a, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)

        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch



    def InfoNCE_logitsr21no(self, f_a, f_1, f_2, f_3, f_4, tau):
        # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

        # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        #pos1 = torch.mm(f_a, self.Wr21)
        pos1 = torch.mm(f_a, f_1.transpose(1, 0)) / tau
       # neg1 = torch.mm(f_a, self.Wr21)
        neg1 = torch.mm(f_a, f_2.transpose(1, 0)) / tau
        #neg2 = torch.mm(f_a, self.Wr21)
        neg2 = torch.mm(f_a, f_3.transpose(1, 0)) / tau
        #neg3 = torch.mm(f_a, self.Wr21)
        neg3 = torch.mm(f_a, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
        # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)

        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsp214no( self, f_a, f_1, f_2, f_3, f_4, tau):
            # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

            # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        #pos1 = torch.mm(f_a, self.Wp214)
        pos1 = torch.mm(f_a, f_1.transpose(1, 0)) / tau
       # neg1 = torch.mm(f_a, self.Wp214)
        neg1 = torch.mm(f_a, f_2.transpose(1, 0)) / tau
      #  neg2 = torch.mm(f_a, self.Wp214)
        neg2 = torch.mm(f_a, f_3.transpose(1, 0)) / tau
       # neg3 = torch.mm(f_a, self.Wp214)
        neg3 = torch.mm(f_a, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
            # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)

        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsr214no(self, f_a, f_1, f_2, f_3, f_4, tau):
            # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

            # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
       # pos1 = torch.mm(f_a, self.Wr214)
        pos1 = torch.mm(f_a, f_1.transpose(1, 0)) / tau
     #   neg1 = torch.mm(f_a, self.Wr214)
        neg1 = torch.mm(f_a, f_2.transpose(1, 0)) / tau
      #  neg2 = torch.mm(f_a, self.Wr214)
        neg2 = torch.mm(f_a, f_3.transpose(1, 0)) / tau
      #  neg3 = torch.mm(f_a, self.Wr214)
        neg3 = torch.mm(f_a, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
            # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)

        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsp2143no( self, f_a, f_1, f_2, f_3, f_4, tau):
            # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

            # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        #pos1 = torch.mm(f_a, self.Wp214)
        pos1 = torch.mm(f_a, f_1.transpose(1, 0)) / tau
       # neg1 = torch.mm(f_a, self.Wp214)
        neg1 = torch.mm(f_a, f_2.transpose(1, 0)) / tau
      #  neg2 = torch.mm(f_a, self.Wp214)
        neg2 = torch.mm(f_a, f_3.transpose(1, 0)) / tau
       # neg3 = torch.mm(f_a, self.Wp214)
        neg3 = torch.mm(f_a, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
            # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)

        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsr2143no(self, f_a, f_1, f_2, f_3, f_4, tau):
            # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

            # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
       # pos1 = torch.mm(f_a, self.Wr214)
        pos1 = torch.mm(f_a, f_1.transpose(1, 0)) / tau
     #   neg1 = torch.mm(f_a, self.Wr214)
        neg1 = torch.mm(f_a, f_2.transpose(1, 0)) / tau
      #  neg2 = torch.mm(f_a, self.Wr214)
        neg2 = torch.mm(f_a, f_3.transpose(1, 0)) / tau
      #  neg3 = torch.mm(f_a, self.Wr214)
        neg3 = torch.mm(f_a, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
            # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)

        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch



    def InfoNCE_logitsp2144no( self, f_a, f_1, f_2, f_3, f_4, tau):
            # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

            # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
        #pos1 = torch.mm(f_a, self.Wp214)
        pos1 = torch.mm(f_a, f_1.transpose(1, 0)) / tau
       # neg1 = torch.mm(f_a, self.Wp214)
        neg1 = torch.mm(f_a, f_2.transpose(1, 0)) / tau
      #  neg2 = torch.mm(f_a, self.Wp214)
        neg2 = torch.mm(f_a, f_3.transpose(1, 0)) / tau
       # neg3 = torch.mm(f_a, self.Wp214)
        neg3 = torch.mm(f_a, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
            # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)

        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch

    def InfoNCE_logitsr2144no(self, f_a, f_1, f_2, f_3, f_4, tau):
            # Normalize the feature representations
        f_a = F.normalize(f_a, p=2, dim=-1)

        f_1 = F.normalize(f_1, p=2, dim=-1)
        f_2 = F.normalize(f_2, p=2, dim=-1)
        f_3 = F.normalize(f_3, p=2, dim=-1)
        f_4 = F.normalize(f_4, p=2, dim=-1)

            # pos = torch.mm(f_a, f_a.transpose(1, 0))/tau
       # pos1 = torch.mm(f_a, self.Wr214)
        pos1 = torch.mm(f_a, f_1.transpose(1, 0)) / tau
     #   neg1 = torch.mm(f_a, self.Wr214)
        neg1 = torch.mm(f_a, f_2.transpose(1, 0)) / tau
      #  neg2 = torch.mm(f_a, self.Wr214)
        neg2 = torch.mm(f_a, f_3.transpose(1, 0)) / tau
      #  neg3 = torch.mm(f_a, self.Wr214)
        neg3 = torch.mm(f_a, f_4.transpose(1, 0)) / tau

        L_1 = pos1
        Ll_1 = torch.exp(torch.diag(L_1, 0))
            # sum_p = torch.log(Ll_0+Ll_1+Ll_2+Ll_3+Ll_4)
        neg1 = torch.sum(torch.exp(neg1), dim=0)

        neg2 = torch.sum(torch.exp(neg2), dim=0)
        neg3 = torch.sum(torch.exp(neg3), dim=0)

        sum_p1 = torch.log(Ll_1)
        negative1 = torch.log(neg1 + Ll_1 + neg2 + neg3)
        loss_batch1 = sum_p1 - negative1

        loss_batch = (loss_batch1)

        return -loss_batch





    def forward(self, x_b2_, x_b1_, x_b1, x_b2, x_b, x_a, x_m, c_b2_, c_b1_, c_b1, c_b2, c_b, c_a, c_middle,x_a2_, x_a1_,c_a2_, c_a1_,epo):
        with torch.no_grad():
            x_b2_1 = x_b2_[:,0:16].contiguous()
            c_b2_1 = c_b2_[:,0:16].contiguous()
            x_b2_2 = x_b2_[:,16:32].contiguous()
            c_b2_2 = c_b2_[:,16:32].contiguous()
            dis_21_1, dis_22_1 = self.int_model_1.encoder(x_b2_1, c_b2_1)
            z_1_1 = dis_21_1.mean
            z_1_2 = dis_22_1.mean

            dis_21_1, dis_22_1 = self.int_model_1.encoder(x_b2_2, c_b2_2)
            z_2_1 = dis_21_1.mean
            z_2_2 = dis_22_1.mean

            x_b1_1 = x_b1_[:,0:16].contiguous()
            c_b1_1 = c_b1_[:,0:16].contiguous()
            x_b1_2 = x_b1_[:,16:32].contiguous()
            c_b1_2 = c_b1_[:,16:32].contiguous()

            dis_21_2, dis_22_2 = self.int_model_1.encoder(x_b1_1, c_b1_1)
            z_3_1 = dis_21_2.mean
            z_3_2 = dis_22_2.mean

            dis_21_2, dis_22_2 = self.int_model_1.encoder(x_b1_2, c_b1_2)
            z_4_1 = dis_21_2.mean
            z_4_2 = dis_22_2.mean

            x_b5_1 = x_b1[:,0:16].contiguous()
            c_b5_1 = c_b1[:,0:16].contiguous()
            x_b6_2 = x_b1[:,16:32].contiguous()
            c_b6_2 = c_b1[:,16:32].contiguous()

            dis_21, dis_22 = self.int_model_1.encoder(x_b5_1,c_b5_1)
            z_5_1 = dis_21.mean
            z_5_2 = dis_22.mean
            dis_21, dis_22 = self.int_model_1.encoder(x_b6_2,c_b6_2)
            z_6_1 = dis_21.mean
            z_6_2 = dis_22.mean

            x_b7_1 = x_b2[:,0:16].contiguous()
            c_b7_1 = c_b2[:,0:16].contiguous()
            x_b8_2 = x_b2[:,16:32].contiguous()
            c_b8_2 = c_b2[:,16:32].contiguous()


            dis_21, dis_22 = self.int_model_1.encoder(x_b7_1,c_b7_1)
            z_7_1 = dis_21.mean
            z_7_2 = dis_22.mean
            dis_21, dis_22 = self.int_model_1.encoder(x_b8_2,c_b8_2)
            z_8_1 = dis_21.mean
            z_8_2 = dis_22.mean

            x_b9_1 = x_m[:,0:16].contiguous()
            c_b9_1 = c_middle[:,0:16].contiguous()
            x_b10_2 = x_m[:,16:32].contiguous()
            c_b10_2 = c_middle[:,16:32].contiguous()
            x_b11_1= x_m[:,32:48].contiguous()
            c_b11_1 = c_middle[:,32:48].contiguous()
            x_b12_1 = x_m[:,48:64].contiguous()
            c_b12_1 = c_middle[:,48:64].contiguous()

            dis_1, dis_2 = self.int_model_1.encoder(x_b9_1, c_b9_1)
            z_m1_1 = dis_1.mean
            z_m1_2 = dis_2.mean

            dis_1, dis_2 = self.int_model_1.encoder(x_b10_2, c_b10_2)
            z_m2_1 = dis_1.mean
            z_m2_2 = dis_2.mean

            dis_1, dis_2 = self.int_model_1.encoder(x_b11_1, c_b11_1)
            z_m3_1 = dis_1.mean
            z_m3_2 = dis_2.mean

            dis_1, dis_2 = self.int_model_1.encoder(x_b12_1, c_b12_1)
            z_m4_1 = dis_1.mean
            z_m4_2 = dis_2.mean




            x_a_1 = x_a2_[:,0:16].contiguous()
            c_a_1 = c_a2_[:,0:16].contiguous()
            x_a_2 = x_a2_[:,16:32].contiguous()
            c_a_2 = c_a2_[:,16:32].contiguous()
            dis_21_1, dis_22_1 = self.int_model_1.encoder(x_a_1, c_a_1)
            z_11_1 = dis_21_1.mean
            z_11_2 = dis_22_1.mean

            dis_21_1, dis_22_1 = self.int_model_1.encoder(x_a_2, c_a_2)
            z_12_1 = dis_21_1.mean
            z_12_2 = dis_22_1.mean

            x_a1_1 = x_a1_[:,0:16].contiguous()
            c_a1_1 = c_a1_[:,0:16].contiguous()
            x_a1_2 = x_a1_[:,16:32].contiguous()
            c_a1_2 = c_a1_[:,16:32].contiguous()

            dis_21_2, dis_22_2 = self.int_model_1.encoder(x_a1_1, c_a1_1)
            z_13_1 = dis_21_2.mean
            z_13_2 = dis_22_2.mean

            dis_21_2, dis_22_2 = self.int_model_1.encoder(x_a1_2, c_a1_2)
            z_14_1 = dis_21_2.mean
            z_14_2 = dis_22_2.mean






            dis_81, dis_82 = self.int_model_8.encoder(x_b, c_b)
           # z8_1 = dis_81.rsample()
            #z8_2 = dis_82.rsample()
            z8_1 = dis_81.mean
            z8_2 = dis_82.mean

            dis_83, dis_84 = self.int_model_4.encoder(x_a, c_a)

            z8_3 = dis_83.mean
            z8_4 = dis_84.mean


        z8_18 = z8_1.unsqueeze(1)
        z8_28 = z8_2.unsqueeze(1)
        z8_38 = z8_3.unsqueeze(1)
        z8_48 = z8_4.unsqueeze(1)
        z_p1 = z_1_1.unsqueeze(1)
        z_p2 = z_2_1.unsqueeze(1)
        z_p3 = z_3_1.unsqueeze(1)
        z_p4 = z_4_1.unsqueeze(1)
        z_p5 = z_5_1.unsqueeze(1)
        z_p6 = z_6_1.unsqueeze(1)
        z_p7 = z_7_1.unsqueeze(1)
        z_p8 = z_8_1.unsqueeze(1)

        z_r1 = z_1_2.unsqueeze(1)
        z_r2 = z_2_2.unsqueeze(1)
        z_r3 = z_3_2.unsqueeze(1)
        z_r4 = z_4_2.unsqueeze(1)
        z_r5 = z_5_2.unsqueeze(1)
        z_r6 = z_6_2.unsqueeze(1)
        z_r7 = z_7_2.unsqueeze(1)
        z_r8 = z_8_2.unsqueeze(1)


        z_p11 = z_11_1.unsqueeze(1)
        z_p12 = z_12_1.unsqueeze(1)
        z_p13 = z_13_1.unsqueeze(1)
        z_p14 = z_14_1.unsqueeze(1)


        z_r11 = z_11_2.unsqueeze(1)
        z_r12 = z_12_2.unsqueeze(1)
        z_r13 = z_13_2.unsqueeze(1)
        z_r14 = z_14_2.unsqueeze(1)





        z_m11 = z_m1_1.unsqueeze(1).contiguous()
        z_m21 = z_m2_1.unsqueeze(1).contiguous()
        z_m12 = z_m1_2.unsqueeze(1).contiguous()
        z_m22 = z_m2_2.unsqueeze(1).contiguous()

        z_m31 = z_m3_1.unsqueeze(1).contiguous()
        z_m32 = z_m3_2.unsqueeze(1).contiguous()

        z_m41 = z_m4_1.unsqueeze(1).contiguous()
        z_m42 = z_m4_2.unsqueeze(1).contiguous()



        z_pp_lr = torch.cat((z_p1, z_p2, z_p3, z_p4, z_p5, z_p6, z_p7, z_p8), 1)
        z_rr_lr = torch.cat((z_r1,z_r2,z_r3,z_r4,z_r5,z_r6,z_r7,z_r8), 1)
        z_pp_rl = torch.cat((z_p14, z_p13, z_p12, z_p11), 1)
        z_rr_rl = torch.cat((z_r14, z_r13, z_r12, z_r11), 1)


        h_p,h_r = [],[]
        h_p1, h_r1 = [], []

        zp_, zr_ = self.preddrl(z_pp_rl, z_rr_rl, z8_1, z8_2,4,4)
        a = 5
        if  epo <0:
            for u in range(11):
                if u < 7:
                    zp, h_p,h_p1, zr, h_r,h_r1 = self.preddlr(z_pp_lr[:, u, :], z_rr_lr[:, u, :], z8_3, z8_4, 8, 4, h_p,h_p1, h_r,h_r1, u)
                if u == 7:
                    zp, h_p,h_p1, zr, h_r,h_r1 = self.preddlr(z_pp_lr[:, u, :], z_rr_lr[:, u, :], z8_3,z8_4, 8, 4, h_p,h_p1, h_r,h_r1, u)
                    print(zp.size())
                    zp_m1 = torch.cat((zp, zp_[:, 3, :]), 1)
                    zr_m1 = torch.cat((zr, zr_[:, 3, :]), 1)
                    zpp1 = self.mid_linearp(zp_m1)
                    zrr1 = self.mid_linearr(zr_m1)
                    recon_rhythm9 = self.int_model_1_.rhythm_decoder(zrr1, x_b9_1)
                    recon9 = self.int_model_1_.final_decoder(zpp1, recon_rhythm9, c_b9_1, x_b9_1)
                if u == 8:
                    zp, h_p,h_p1, zr, h_r,h_r1 = self.preddlr(z_pp_lr[:, u, :], z_rr_lr[:, u, :], z8_3, z8_4, 8, 4, h_p,h_p1, h_r,h_r1, u)
                    zp_m2 = torch.cat((zp, zp_[:, 2, :]), 1)
                    zr_m2 = torch.cat((zr, zr_[:, 2, :]), 1)
                    zpp2 = self.mid_linearp(zp_m2)
                    zrr2 = self.mid_linearr(zr_m2)
                    recon_rhythm10 = self.int_model_1_.rhythm_decoder(zrr2, x_b10_2)
                    recon10 = self.int_model_1_.final_decoder(zpp2, recon_rhythm10, c_b10_2, x_b10_2)
                if u == 9:
                    zp, h_p,h_p1, zr, h_r,h_r1 = self.preddlr(z_pp_lr[:, u, :], z_rr_lr[:, u, :], z8_3, z8_4, 8, 4, h_p,h_p1, h_r,h_r1, u)
                    zp_m3 = torch.cat((zp, zp_[:, 1, :]), 1)
                    zr_m3 = torch.cat((zr, zr_[:, 1, :]), 1)
                    zpp3 = self.mid_linearp(zp_m3)
                    zrr3 = self.mid_linearr(zr_m3)
                    recon_rhythm11 = self.int_model_1_.rhythm_decoder(zrr3, x_b11_1)
                    recon11 = self.int_model_1_.final_decoder(zpp3, recon_rhythm11, c_b11_1, x_b11_1)
                if u == 10:
                    zp, h_p,h_p1, zr, h_r,h_r1 = self.preddlr(z_pp_lr[:, u, :], z_rr_lr[:, u, :], z8_3, z8_4, 8, 4, h_p,h_p1, h_r,h_r1, u)
                    zp_m4 = torch.cat((zp, zp_[:, 0, :]), 1)
                    zr_m4 = torch.cat((zr, zr_[:, 0, :]), 1)
                    zpp4 = self.mid_linearp(zp_m4)
                    zrr4 = self.mid_linearr(zr_m4)
                    recon_rhythm12 = self.int_model_1_.rhythm_decoder(zrr4, x_b12_1)
                    recon12 = self.int_model_1_.final_decoder(zpp4, recon_rhythm12, c_b12_1, x_b12_1)
        else:
            for u in range(11):
                if u < 7:
                    zp, h_p,h_p1, zr, h_r,h_r1 = self.preddlr(z_pp_lr[:, u, :], z_rr_lr[:, u, :], z8_3, z8_4, 8, 4, h_p,h_p1, h_r,h_r1, u)
                elif u == 7:
                    zp, h_p,h_p1, zr, h_r,h_r1 = self.preddlr(z_pp_lr[:, u, :], z_rr_lr[:, u, :], z8_3,z8_4, 8, 4, h_p,h_p1, h_r,h_r1, u)
                    print(zp.size())
                    zp_m1 = torch.cat((zp, zp_[:, 3, :]), 1)
                    zr_m1 = torch.cat((zr, zr_[:, 3, :]), 1)
                    zpp1 = self.mid_linearp(zp_m1)
                    zrr1 = self.mid_linearr(zr_m1)
                elif u == 8:
                    recon_rhythm9 = self.int_model_1_.rhythm_decoder(zrr1, x_b9_1)
                    recon9 = self.int_model_1_.final_decoder(zpp1, recon_rhythm9, c_b9_1, x_b9_1)
                    recon9_ =  F.gumbel_softmax(recon9.contiguous(), tau=1, hard=True)
                    with torch.no_grad():
                        dis_1, dis_2 = self.int_model_1.encoder(recon9_, c_b9_1)
                        z_p_ = dis_1.mean.contiguous()
                        z_r_ = dis_2.mean.contiguous()
                    zp, h_p,h_p1, zr, h_r,h_r1 = self.preddlr(z_p_,z_r_,z8_3, z8_4, 8, 4,h_p,h_p1, h_r,h_r1, u)
                    zp_m2 = torch.cat((zp, zp_[:, 2, :]), 1)
                    zr_m2 = torch.cat((zr, zr_[:, 2, :]), 1)
                    zpp2 = self.mid_linearp(zp_m2)
                    zrr2 = self.mid_linearr(zr_m2)
                elif u == 9:
                    recon_rhythm10 = self.int_model_1_.rhythm_decoder(zrr2, x_b10_2)
                    recon10 = self.int_model_1_.final_decoder(zpp2, recon_rhythm10, c_b10_2, x_b10_2)
                    recon10_ = F.gumbel_softmax(recon10.contiguous(), tau=1, hard=True)
                    with torch.no_grad():
                        dis_1, dis_2 = self.int_model_1.encoder(recon10_, c_b10_2)
                        z_p_ = dis_1.mean.contiguous()
                        z_r_ = dis_2.mean.contiguous()
                    zp, h_p,h_p1, zr, h_r,h_r1 = self.preddlr(z_p_,z_r_, z8_3, z8_4, 8, 4, h_p,h_p1, h_r,h_r1, u)
                    zp_m3 = torch.cat((zp, zp_[:, 1, :]), 1)
                    zr_m3 = torch.cat((zr, zr_[:, 1, :]), 1)
                    zpp3 = self.mid_linearp(zp_m3)
                    zrr3 = self.mid_linearr(zr_m3)
                elif u == 10:
                    recon_rhythm11 = self.int_model_1_.rhythm_decoder(zrr3, x_b11_1)
                    recon11 = self.int_model_1_.final_decoder(zpp3, recon_rhythm11, c_b11_1, x_b11_1)
                    recon11_ = F.gumbel_softmax(recon11.contiguous(), tau=1, hard=True)
                    with torch.no_grad():
                        dis_1, dis_2 = self.int_model_1.encoder(recon11_, c_b11_1)
                        z_p_ = dis_1.mean.contiguous()
                        z_r_ = dis_2.mean.contiguous()
                    zp, h_p,h_p1, zr, h_r,h_r1 = self.preddlr(z_p_,z_r_,  z8_3, z8_4, 8, 4, h_p,h_p1, h_r,h_r1, u)
                    zp_m4 = torch.cat((zp, zp_[:, 0, :]), 1)
                    zr_m4 = torch.cat((zr, zr_[:, 0, :]), 1)
                    zpp4 = self.mid_linearp(zp_m4)
                    zrr4 = self.mid_linearr(zr_m4)
                    recon_rhythm12 = self.int_model_1_.rhythm_decoder(zrr4, x_b12_1)
                    recon12 = self.int_model_1_.final_decoder(zpp4, recon_rhythm12, c_b12_1, x_b12_1)










        cpcpm1 = self.InfoNCE_logitsp21no(zpp1, z_m1_1, z_m1_2, z_11_1, z_12_1, 1)
        cpcrm1 = self.InfoNCE_logitsr21no(zrr1, z_m1_2, z_m1_1, z_11_2, z_12_2, 1)

        cpcpm2= self.InfoNCE_logitsp214no(zpp2, z_m2_1, z_m2_2, z_11_1, z_12_1, 1)
        cpcrm2 = self.InfoNCE_logitsr214no(zrr2, z_m2_2, z_m2_1, z_11_1, z_12_1, 1)

        cpcpm3= self.InfoNCE_logitsp2143no(zpp3, z_m3_1, z_m3_2, z_11_1, z_12_1, 1)
        cpcrm3 = self.InfoNCE_logitsr2143no(zrr3, z_m3_2, z_m3_1, z_11_1, z_12_1, 1)

        cpcpm4= self.InfoNCE_logitsp2144no(zpp4, z_m4_1, z_m4_2, z_11_1, z_12_1, 1)
        cpcrm4 = self.InfoNCE_logitsr2144no(zrr4, z_m4_2, z_m4_1, z_11_1, z_12_1, 1)


        out = (recon9, recon_rhythm9,recon10, recon_rhythm10,recon11, recon_rhythm11,recon12, recon_rhythm12,dis_1.mean,dis_1.stddev,dis_2.mean,dis_2.stddev,cpcpm1,cpcrm1,cpcpm2,cpcrm2,cpcpm3,cpcrm3,cpcpm4,cpcrm4)
        return out
