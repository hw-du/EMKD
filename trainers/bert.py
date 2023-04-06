from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks


import torch
import torch.nn as nn
import torch.nn.functional as F

import random 

class NTXENTloss(nn.Module):

    def __init__(self, args, device,temperature=1.):
        super(NTXENTloss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.projection_dim = args.bert_hidden_units
        self.device = device
        self.w1 = nn.Linear(self.projection_dim, self.projection_dim, bias=False).to(self.device)
        self.bn1 = nn.BatchNorm1d(self.projection_dim).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.w2 = nn.Linear(self.projection_dim, self.projection_dim, bias=False).to(self.device)
        self.bn2 = nn.BatchNorm1d(self.projection_dim, affine=False).to(self.device)
        #self.cossim = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def project(self, h):
        return self.bn2(self.w2(self.relu(self.bn1(self.w1(h)))))
    def cosinesim(self,h1,h2):
        h = torch.matmul(h1, h2.T)
        h1_norm2 = h1.pow(2).sum(dim=-1).sqrt().view(h.shape[0],1)
        h2_norm2 = h2.pow(2).sum(dim=-1).sqrt().view(1,h.shape[0])
        return h/(h1_norm2@h2_norm2)
    def forward(self, h1, h2,calcsim='cosine'):
        b = h1.shape[0]
        if self.args.projectionhead:
            z1, z2 = self.project(h1.view(b*self.args.bert_max_len,self.args.bert_hidden_units)), self.project(h2.view(b*self.args.bert_max_len,self.args.bert_hidden_units))
        else:
            z1, z2 = h1, h2
        z1 = z1.view(b, self.args.bert_max_len*self.args.bert_hidden_units)
        z2 = z2.view(b, self.args.bert_max_len*self.args.bert_hidden_units)
        if calcsim=='dot':
            sim11 = torch.matmul(z1, z1.T) / self.temperature
            sim22 = torch.matmul(z2, z2.T) / self.temperature
            sim12 = torch.matmul(z1, z2.T) / self.temperature
        elif calcsim=='cosine':
            sim11 = self.cosinesim(z1, z1) / self.temperature
            sim22 = self.cosinesim(z2, z2) / self.temperature
            sim12 = self.cosinesim(z1, z2) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
        targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
        ntxentloss = self.criterion(raw_scores, targets)
        return ntxentloss

class ICL(nn.Module):

    def __init__(self, args, device,temperature=1.):
        super(ICL, self).__init__()
        self.args = args
        self.temperature = temperature
        self.projection_dim = args.bert_hidden_units
        self.device = device

        self.cossim = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss().to(self.device)


    def cosinesim(self,h1,h2):
        h = torch.matmul(h1, h2.T)
        h1_norm2 = h1.pow(2).sum(dim=-1).sqrt().view(h.shape[0],1)
        h2_norm2 = h2.pow(2).sum(dim=-1).sqrt().view(1,h.shape[0])
        return h/(h1_norm2@h2_norm2)
    def forward(self, anchor, positives_list):
        b = anchor.shape[0]
        anchor = anchor.view(b,-1)
        in_batch_negative_score = self.cosinesim(anchor, anchor) / self.temperature
        in_batch_negative_score = in_batch_negative_score*(1-(torch.eye(b).to(self.device)))
        anchor_positive_sim_score_list = []
        for positive in positives_list:
            positive = positive.view(b,-1)
            anchor_positive_sim_score_list.append(self.cossim(anchor,positive)/self.temperature)
        anchor_positive_score = torch.stack(anchor_positive_sim_score_list, dim = 1)#[batch_size,M]
        nominator = torch.logsumexp(anchor_positive_score, -1)
        #diag_mask = 1.0 - torch.eye(b).to(self.device)
        #denominator = torch.log(torch.sum(torch.exp(in_batch_negative_score)*diag_mask,dim=-1))
        denominator = torch.logsumexp(in_batch_negative_score, -1)

        loss =  torch.sum(-nominator + denominator)/b

        return loss
class CCL(nn.Module):

    def __init__(self, args, device,temperature=1.):
        super(CCL, self).__init__()
        self.args = args
        self.temperature = temperature
        self.projection_dim = args.bert_hidden_units
        self.device = device

        self.cossim = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss().to(self.device)


    def cosinesim(self,h1,h2):
        h = torch.matmul(h1, h2.T)
        h1_norm2 = h1.pow(2).sum(dim=-1).sqrt().view(h.shape[0],1)
        h2_norm2 = h2.pow(2).sum(dim=-1).sqrt().view(1,h.shape[0])
        return h/(h1_norm2@h2_norm2)
    def forward(self, anchorx, anchory, positives_list_y):
        #anchor: [batch_size,seq_len,hidden]
        #positives_list: [[batch_size,seq_len,hidden],...]
        b = anchorx.shape[0]
        anchorx = anchorx.view(b,-1)
        anchory = anchory.view(b,-1)
        in_batch_negative_score = self.cosinesim(anchorx, anchory) / self.temperature
        in_batch_negative_score = in_batch_negative_score*(1-(torch.eye(b).to(self.device)))
        anchor_positive_sim_score_list = []
        for positive in positives_list_y:
            positive = positive.view(b,-1)
            anchor_positive_sim_score_list.append(self.cossim(anchorx,positive)/self.temperature)
        anchor_positive_score = torch.stack(anchor_positive_sim_score_list, dim = 1)#[batch_size,M]
        #print('anchor_positive_score_shape:',anchor_positive_score.shape)
        nominator = torch.logsumexp(anchor_positive_score, -1)
        #diag_mask = 1.0 - torch.eye(b).to(self.device)
        #denominator = torch.log(torch.sum(torch.exp(in_batch_negative_score)*diag_mask,dim=-1))
        denominator = torch.logsumexp(in_batch_negative_score, -1)

        loss =  torch.sum(-nominator + denominator)/b

        return loss
class ICL_crossentropy(nn.Module):

    def __init__(self, args, device,temperature=1.):
        super(ICL, self).__init__()
        self.args = args
        self.temperature = temperature
        self.projection_dim = args.bert_hidden_units
        self.device = device

        self.cossim = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss().to(self.device)


    def cosinesim(self,h1,h2):
        h = torch.matmul(h1, h2.T)
        h1_norm2 = h1.pow(2).sum(dim=-1).sqrt().view(h.shape[0],1)
        h2_norm2 = h2.pow(2).sum(dim=-1).sqrt().view(1,h.shape[0])
        return h/(h1_norm2@h2_norm2)
    def forward(self, anchor, positives_list):
        b = anchor.shape[0]
        anchor = anchor.view(b,-1)
        neg = self.cosinesim(anchor, anchor) / self.temperature#[batch_size,batch_size]
        d = neg.shape[-1]
        neg[..., range(d), range(d)] = float('-inf')
        logits_list = []#M*[batch_size,batch_size+1]
        for positive in positives_list:
            positive = positive.view(b,-1)
            pos_score = self.cossim(anchor,positive) / self.temperature
            logits = torch.cat((pos_score.reshape(b,1), neg), dim=1)
            logits_list.append(logits)
        logits_all = torch.cat(logits_list, dim=0)
        labels = torch.zeros(logits_all.shape[0], dtype = torch.long, device = self.device)
        loss = self.criterion(logits_all, labels)
        return loss
    
class CCL_crossentropy(nn.Module):

    def __init__(self, args, device,temperature=1.):
        super(CCL, self).__init__()
        self.args = args
        self.temperature = temperature
        self.projection_dim = args.bert_hidden_units
        self.device = device

        self.cossim = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss().to(self.device)


    def cosinesim(self,h1,h2):
        h = torch.matmul(h1, h2.T)
        h1_norm2 = h1.pow(2).sum(dim=-1).sqrt().view(h.shape[0],1)
        h2_norm2 = h2.pow(2).sum(dim=-1).sqrt().view(1,h.shape[0])
        return h/(h1_norm2@h2_norm2)
    def forward(self, anchorx, anchory, positives_list_y):
        b = anchorx.shape[0]
        anchorx = anchorx.view(b,-1)
        anchory = anchory.view(b,-1)
        neg = self.cosinesim(anchorx, anchory) / self.temperature#[batch_size,batch_size]
        d = neg.shape[-1]
        neg[..., range(d), range(d)] = float('-inf')
        logits_list = []#M*[batch_size,batch_size+1]
        for positive in positives_list_y:
            positive = positive.view(b,-1)
            pos_score = self.cossim(anchorx,positive) / self.temperature
            logits = torch.cat((pos_score.reshape(b,1), neg), dim=1)
            logits_list.append(logits)
        logits_all = torch.cat(logits_list, dim=0)
        labels = torch.zeros(logits_all.shape[0], dtype = torch.long, device = self.device)
        loss = self.criterion(logits_all, labels)
        return loss
class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader,  export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        
        #self.NTXENTloss = NTXENTloss(args,self.device,self.args.tau)
        self.ICL = ICL(args,self.device,self.args.tau)
        self.CCL = CCL(args,self.device,self.args.tau)
        self.bce = nn.BCEWithLogitsLoss()
        
        self.lambda_=args.lambda_
        self.mu = args.mu
        self.theta = 0
        self.args = args
        
        random.seed(0)
    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass


    def compute_kl_div(self, p, q):

        p_loss = F.kl_div(F.log_softmax(p/self.args.tau, dim=-1), F.softmax(q.detach()/self.args.tau, dim=-1), reduction='batchmean')
        q_loss = F.kl_div(F.log_softmax(q/self.args.tau, dim=-1), F.softmax(p.detach()/self.args.tau, dim=-1), reduction='batchmean')
        loss = (p_loss + q_loss) / 2
        return loss
    def calculate_loss(self, batch):
        attr_tokens = batch[0]
        num_positive=(len(batch)-1)//2


        pairs = [[] for _ in range(num_positive)]
            
        main_loss = 0
        for i in range(num_positive):

            seqs=batch[2*i+1]
            labels=batch[2*i+2]
            recoutputs,c_is = self.model(seqs, output_type = 'token')
            pairs[i] = c_is
            for n in range(self.args.N):
                logits_k = recoutputs[n]
                main_loss += self.ce(logits_k.view(-1, logits_k.size(-1)), labels.view(-1))
            active_index = torch.where(labels>0)
            for a in range(self.args.N):
                for b in range(self.args.N):
                    if a>b:
                        logits_kl_div = self.compute_kl_div(recoutputs[a][active_index],recoutputs[b][active_index])
                        main_loss += self.mu*logits_kl_div

        attr_recoutputs,anchors = self.model(attr_tokens,output_type = 'attributes')

        #indices = list(range(0,num_positive))
        #N: the index of the network
        ICL=0 
        CCL=0
        for n in range(self.args.N):
            attr_logits_k=attr_recoutputs[n]
            h = attr_logits_k.size(-1)
            attr_logits_k = (attr_logits_k[active_index]).view(-1,h)
            attr_labels = (self.args.attribute_one_hot_label_matrix[attr_tokens]).to(self.args.device)
            #attr_labels = (self.args.attribute_one_hot_label_matrix[attr_tokens.cpu()]).to(self.args.device)#use this line in case the device setting goes wrong
            attr_loss = self.bce(attr_logits_k, (attr_labels[active_index]).view(-1,self.args.num_attributes+1))
            main_loss += attr_loss
            ICL += self.ICL(anchors[n],[pairs[positive_index][n] for positive_index in range(num_positive)])#this model
            #for a in range(num_positive):                        
                #ICL += self.NTXENTloss(pairs[a][m],anchors[m], calcsim=self.args.calcsim)
            #cl_loss += self.NTXENTloss(pairs[a][m],anchors[m], calcsim=self.args.calcsim)#this model

            for u in range(self.args.N):
                if u!=n:
                    CCL += self.CCL(anchors[n],anchors[u],[pairs[positive_index][u] for positive_index in range(num_positive)])#other model
                    #for a in range(num_positive): 
                    #    CCL += self.NTXENTloss(pairs[a][m],anchors[u], calcsim=self.args.calcsim)#other model
        cl_loss = ICL+CCL        
        main_loss += self.lambda_*cl_loss 

        return main_loss

    def calculate_loss_sample(self, batch):
        attr_tokens = batch[0]
        num_positive=(len(batch)-1)//2
        pairs = [[] for _ in range(num_positive)]
            
        main_loss = 0
        cl_loss=0
        for i in range(num_positive):

            seqs=batch[2*i+1]
            labels=batch[2*i+2]
            recoutputs,c_is = self.model(seqs, output_type = 'token')
            pairs[i] = c_is
            for n in range(self.args.N):
                logits_k = recoutputs[n]
                main_loss += self.ce(logits_k.view(-1, logits_k.size(-1)), labels.view(-1))
            active_index = torch.where(labels>0)
            for a in range(self.args.N):
                for b in range(self.args.N):
                    if a>b:
                        logits_kl_div = self.compute_kl_div(recoutputs[a][active_index],recoutputs[b][active_index])
                        main_loss += self.lambda_*logits_kl_div

        attr_recoutputs,anchors = self.model(attr_tokens,output_type = 'attributes')

        indices = list(range(0,num_positive))
        for n in range(self.args.N):
            attr_logits_k=attr_recoutputs[n]
            h = attr_logits_k.size(-1)
            attr_logits_k = (attr_logits_k[active_index]).view(-1,h)
            attr_labels = (self.args.attribute_one_hot_label_matrix[attr_tokens]).to(self.args.device)
            attr_loss = self.bce(attr_logits_k, (attr_labels[active_index]).view(-1,self.args.num_attributes+1))
            main_loss += attr_loss
            #a,b = random.sample(indices,2)
            a = random.choice(indices)
            cl_loss += self.NTXENTloss(pairs[a][n],anchors[n], calcsim=self.args.calcsim)#this model
            for u in range(self.args.N):
                if u!=n:
                    x = random.choice(indices)
                    cl_loss += self.NTXENTloss(pairs[x][n],anchors[u], calcsim=self.args.calcsim)#other model
                    

            
        num_main_loss = main_loss.detach().data.item()
        num_cl_loss = cl_loss.detach().data.item()
        theta_hat = num_main_loss/(num_main_loss+5*num_cl_loss)
        self.theta = self.alpha*theta_hat+(1-self.alpha)*self.theta
        main_loss += self.theta*cl_loss

        return main_loss

    

    def calculate_metrics(self, batch, metric_type):
        index = batch

        labels = self.args.matrix_label.repeat(index.shape[0],1).to(self.args.device)

        if metric_type == 'validate':
            seqs = self.args.eval_matrix_seq[index].squeeze().to(self.args.device)

            candidates = self.args.eval_matrix_candidate[index].squeeze().to(self.args.device)
        elif metric_type == 'test':
            seqs = self.args.test_matrix_seq[index].squeeze().to(self.args.device)
            candidates = self.args.test_matrix_candidate[index].squeeze().to(self.args.device)


        recoutputs,c_is = self.model(seqs, output_type = 'token')
        scores = None
        for i in range(self.args.N):
            if i == 0:
                scores = recoutputs[i][:, -1, :]
            else:
                scores = scores + recoutputs[i][:, -1, :]
        

        scores[:,0] = -999.999# pad token should not appear in the logits output

        scores = scores.gather(1, candidates)#the whole item set 

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)

        return metrics

