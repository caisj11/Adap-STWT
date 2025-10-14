'''
Author: Caisj
Date: 2024-07-11 19:37:55
LastEditTime: 2025-10-13 17:52:06
'''
# -*- coding: utf-8 -*-
# @Time : 2023/8/15 10:11
# @Author : Caisj
from collections import Counter
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import config
from model.AdapSTWT.channel_attention import ChannelAttention
from model.AdapSTWT.spatial_attention import SpatialAttentionLayer, SpatialAttention
from torch.nn import functional as F
# from fastdtw import fastdtw
from scipy.spatial.distance import euclidean



class GraphLearn(torch.nn.Module):
    def __init__(self, num_nodes, time_step, heads, dropout, graph_emb, device):
        super(GraphLearn, self).__init__()
        self.num_nodes = num_nodes
        self.time_step = time_step
        self.device = device
        self.lamda1 = torch.eye(num_nodes, dtype=torch.float32, device=device)
        self.lamda2 = torch.eye(num_nodes, dtype=torch.float32, device=device)
        self.beta = torch.nn.Parameter(torch.rand(num_nodes, dtype=torch.float32), requires_grad=True)
        self.w1 = torch.nn.Parameter(torch.zeros((num_nodes, graph_emb), dtype=torch.float32), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.zeros((num_nodes, graph_emb), dtype=torch.float32), requires_grad=True)

        # spatial Transformer
        self.conv2d = torch.nn.Conv2d(2, graph_emb, kernel_size=1)
        self.spatial_attention = SpatialAttentionLayer(SpatialAttention(False, attention_dropout=dropout), graph_emb, heads)
        self.time_conv = torch.nn.Sequential(
            torch.nn.Conv2d(graph_emb, graph_emb, kernel_size=(1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(graph_emb, graph_emb, kernel_size=(1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(graph_emb, 12, kernel_size=(self.time_step, 1))
        )
        # fusion weights
        self.softmax = torch.nn.Softmax()
        self.attn = torch.nn.Conv2d(2, 1, kernel_size=1)
        self.graph_attn = torch.nn.Conv2d(2, 1, kernel_size=1)

        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, x, adj_mx, traj_mx):
        # x:[B,T,N,C]
        # ---------------- step1：Local Graph Learning (Macro Graph) ----------------
        # update local adjacency matrix
        local_adj_mx = torch.mm(self.w1, self.w2.T) - torch.mm(self.w2, self.w1.T)
        local_adj_mx = torch.relu(local_adj_mx + self.lamda1)
        # Mask
        one = torch.ones_like(adj_mx)
        mask_matrix = torch.where(adj_mx > 0, one, adj_mx)
        local_adj_mx = torch.mul(local_adj_mx, mask_matrix)
        # fusion with original adjacency matrix
        local_attn = torch.sigmoid(self.attn(torch.stack((local_adj_mx, adj_mx), dim=0).unsqueeze(dim=0)).squeeze())
        local_adj_mx = local_attn * local_adj_mx + (1. - local_attn) * adj_mx
        # normalize
        d = local_adj_mx.sum(dim=1) ** (-0.5)
        local_adj_mx = d.view(-1, 1) * local_adj_mx * d

        # ---------------- step2: Global Graph Learning (Macro Graph) ----------------
        # spatial attention
        x = self.conv2d(x.transpose(1, 3)).transpose(1, 3).transpose(1, 2)
        global_out, _ = self.spatial_attention(x, x, x)
        # considering temporal correlation
        global_out = self.time_conv(global_out.transpose(1, 3)).squeeze(2).transpose(1, 2) # [B,N,C]
        global_adj_mx = torch.einsum('bnc,bvc->nv', [global_out, global_out])
        global_adj_mx = F.normalize(torch.relu(global_adj_mx + self.lamda2), p=1, dim=-1)
        # Mask
        mask_matrix = torch.where(adj_mx > 0, torch.ones_like(adj_mx), adj_mx)
        mask_matrix = torch.abs(mask_matrix - 1)  # mask local connections
        global_adj_mx = torch.mul(global_adj_mx, mask_matrix)
        
        # ---------------- step3：Micro Graph Learning  ----------------
        # the real-time semantic expression relationship of trajectories that have been learned in preprocessing, 
        # with the input being a semantic relationship matrix
        micro_adj_mx = torch.mean(traj_mx.float(), dim=0)
        micro_adj_mx = torch.relu(micro_adj_mx + torch.diag(self.beta))
        d = micro_adj_mx.sum(dim=1) ** (-0.5)
        micro_adj_mx = d.view(-1, 1) * micro_adj_mx * d

        # ---------------- step4: Fusion Macro and Micro Graph ----------------
        macro_adj_mx = torch.relu(local_adj_mx + global_adj_mx)
        # sigmoid fusion
        attn = torch.sigmoid(self.graph_attn(torch.stack((macro_adj_mx, micro_adj_mx), dim=0).unsqueeze(dim=0)).squeeze())
        new_adj_mx = attn * macro_adj_mx + (1. - attn) * micro_adj_mx
        # new_adj_mx = macro_adj_mx
        new_adj_mx = torch.relu(new_adj_mx)

        return new_adj_mx



class ChangeAdj:
    def __init__(self, device):
        self.device = device

    def edge_index_func(self, best_adj_mx):
        node_begin, node_end, edge_weight = [], [], []
        # Traverse adjacency relationship data
        for i in range(best_adj_mx.shape[0]):
            for j in range(best_adj_mx.shape[1]):
                if best_adj_mx[i][j] != 1:
                    node_begin.append(i)
                    node_end.append(j)
                    edge_weight.append(best_adj_mx[i][j])
        edge_index = [node_begin, node_end]
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float, device=self.device)
        return edge_index, edge_weight
    

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.init_weights()
    
    def init_weights(self):
        init_range = 0.5 / self.embedding_dim
        self.center_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, center_words, context_words, negative_words):
        center_embeds = self.center_embeddings(center_words)  # [batch_size, embed_dim]
        context_embeds = self.context_embeddings(context_words)  # [batch_size, embed_dim]
        neg_embeds = self.context_embeddings(negative_words)  # [batch_size, neg_samples, embed_dim]

        pos_score = torch.sum(center_embeds * context_embeds, dim=1)  # [batch_size]
        pos_score = torch.sigmoid(pos_score)

        neg_score = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze(2)  # [batch_size, neg_samples]
        neg_score = torch.sigmoid(-neg_score)
        
        return pos_score, neg_score

class Word2VecTrainer:
    def __init__(self, vector_size=256, window=6, min_count=1, negative=10, epochs=50, lr=0.001, device='cuda'):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.epochs = epochs
        self.lr = lr
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.vocab_size = 0
        self.model = None
        
    def build_vocab(self, sentences):
        for sentence in sentences:
            for word in sentence:
                self.word_counts[str(word)] += 1
        
        filtered_words = [word for word, count in self.word_counts.items() if count >= self.min_count]
        
        self.word2idx = {word: idx for idx, word in enumerate(filtered_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        self.create_negative_sampling_table()
    
    def create_negative_sampling_table(self):
        total_count = sum(self.word_counts.values())
        word_probs = []
        
        for word in self.idx2word.values():
            count = self.word_counts[word]
            prob = (count / total_count) ** 0.75
            word_probs.append(prob)
        
        word_probs = np.array(word_probs)
        word_probs = word_probs / word_probs.sum()
        
        self.negative_sampling_probs = word_probs
    
    def generate_training_data(self, sentences):
        training_data = []
        
        for sentence in sentences:
            sentence_words = [str(word) for word in sentence if str(word) in self.word2idx]
            
            if len(sentence_words) < 2:
                continue
                
            for i, center_word in enumerate(sentence_words):
                start = max(0, i - self.window)
                end = min(len(sentence_words), i + self.window + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_word = sentence_words[j]
                        training_data.append((center_word, context_word))
        
        return training_data
    
    def get_negative_samples(self, batch_size):
        negative_samples = np.random.choice(
            self.vocab_size, 
            size=(batch_size, self.negative), 
            p=self.negative_sampling_probs
        )
        return negative_samples
    
    def train(self, sentences):
        self.build_vocab(sentences)
        
        if self.vocab_size == 0:
            return
        
        self.model = SkipGramModel(self.vocab_size, self.vector_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        training_data = self.generate_training_data(sentences)
        
        if len(training_data) == 0:
            return
        
        batch_size = min(8192, len(training_data))
        
        for epoch in range(self.epochs):
            total_loss = 0
            random.shuffle(training_data)
            
            for i in range(0, len(training_data), batch_size):
                batch_data = training_data[i:i+batch_size]
                
                if len(batch_data) == 0:
                    continue
                
                center_words = [self.word2idx[pair[0]] for pair in batch_data]
                context_words = [self.word2idx[pair[1]] for pair in batch_data]
                
                negative_samples = self.get_negative_samples(len(batch_data))
                
                center_tensor = torch.LongTensor(center_words).to(self.device)
                context_tensor = torch.LongTensor(context_words).to(self.device)
                negative_tensor = torch.LongTensor(negative_samples).to(self.device)

                pos_score, neg_score = self.model(center_tensor, context_tensor, negative_tensor)
                
                pos_loss = -torch.log(pos_score + 1e-8).mean()
                neg_loss = -torch.log(neg_score + 1e-8).sum(dim=1).mean()
                loss = pos_loss + neg_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
    
    def get_vector(self, word):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        word_str = str(word)
        if word_str not in self.word2idx:
            raise KeyError(f"Word '{word_str}' not in vocabulary")
        
        idx = self.word2idx[word_str]
        with torch.no_grad():
            embedding = self.model.center_embeddings.weight[idx].cpu().numpy()
        return embedding

def transfer_probability(traj_mx, nodes_grid, vector_size=100, window=5, min_count=1, 
                        negative=10, epochs=50, lr=0.001, device='cuda'):
    if not traj_mx or len(traj_mx) == 0:
        return np.eye(len(nodes_grid), dtype=np.float32)
    
    if isinstance(traj_mx, list) and len(traj_mx) > 0 and isinstance(traj_mx[0], str):
        traj_mx = [traj_mx]
    
    valid_trajs = [traj for traj in traj_mx if traj and len(traj) > 0]
    
    if not valid_trajs or len(valid_trajs) < 2:
        return np.eye(len(nodes_grid), dtype=np.float32)
    
    trainer = Word2VecTrainer(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        epochs=epochs,
        lr=lr,
        device=device
    )
    
    trainer.train(valid_trajs)
    
    if trainer.model is None:
        return np.eye(len(nodes_grid), dtype=np.float32)
    
    node_embedding = []
    oov_count = 0
    
    for node in nodes_grid:
        try:
            res = trainer.get_vector(str(node))
        except (KeyError, ValueError):
            oov_count += 1
            res = np.zeros(vector_size, dtype=np.float32)
        node_embedding.append(res)
    
    n_nodes = len(node_embedding)
    matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    
    if torch.cuda.is_available() and device == 'cuda':
        embeddings_array = np.array(node_embedding, dtype=np.float32)
        embeddings_tensor = torch.from_numpy(embeddings_array).to(device)
        
        with torch.no_grad():
            embeddings_expanded = embeddings_tensor.unsqueeze(1)  # [n, 1, d]
            embeddings_T = embeddings_tensor.unsqueeze(0)  # [1, n, d]
            
            diff = embeddings_expanded - embeddings_T  # [n, n, d]
            distances = torch.norm(diff, dim=2)  # [n, n]
            matrix = distances.cpu().numpy()
    else:
        for i in range(n_nodes-1):
            for j in range(i+1, n_nodes):
                if (np.array(node_embedding[i]) == 0).all() or (np.array(node_embedding[j]) == 0).all():
                    matrix[i, j] = 0
                else:
                    dist = np.linalg.norm(np.array(node_embedding[i]) - np.array(node_embedding[j]))
                    matrix[i, j] = dist
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[j, i] = matrix[i, j]
    
    std, mean = np.std(matrix), np.mean(matrix)
    
    if std > 0:
        for i in range(len(matrix)):
            matrix[i] = [math.exp(-(x ** 2) / (2 * std ** 2)) if x != 0 else 0 for x in matrix[i]]

    final_matrix = matrix + np.eye(len(matrix), dtype=np.float32)
    
    return final_matrix



