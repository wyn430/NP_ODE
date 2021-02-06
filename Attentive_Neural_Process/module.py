import torch as t
import torch.nn as nn
import math
from torch.distributions import Independent, Normal,MultivariateNormal
from torch.nn import functional as F

class Linear(nn.Module):
    """
    Linear Module
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init='relu'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)

class LatentEncoder(nn.Module):
    """
    Latent Encoder [For prior, posterior]
    """
    def __init__(self, num_hidden, num_latent, input_dim):
        super(LatentEncoder, self).__init__()
        self.input_projection = Linear(input_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.mu = Linear(num_hidden, num_latent)
        self.log_sigma = Linear(num_hidden, num_latent)

    def forward(self, x, y):
        # concat location (x) and value (y)
        encoder_input = t.cat([x,y], dim=-1)
        
        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)
        
        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)
        
        # mean
        hidden = encoder_input.mean(dim=1)
        hidden = t.relu(self.penultimate_layer(hidden))
        
        # get mu and sigma
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)
        
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
#        std = t.exp(0.5 * log_sigma)
#        eps = t.randn_like(std)
        z = sigma.add_(mu)
        
        # return distribution
        return mu, log_sigma, z
    
class DeterministicEncoder(nn.Module):
    """
    Deterministic Encoder [r]
    """
    def __init__(self, num_hidden, num_latent, input_dim, y_dim):
        super(DeterministicEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.input_projection = Linear(input_dim, num_hidden)
        self.context_projection = Linear(input_dim-y_dim, num_hidden)
        self.target_projection = Linear(input_dim-y_dim, num_hidden)

    def forward(self, context_x, context_y, target_x):
        # concat context location (x), context value (y)
        encoder_input = t.cat([context_x,context_y], dim=-1)
        
        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)
        
        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)
        
        # query: target_x, key: context_x, value: representation
        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)
        
        # cross attention layer
        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query)
        
        return query
    
class Decoder(nn.Module):
    """
    Decoder for generation 
    """
    def __init__(self, num_hidden, input_dim, y_dim):
        super(Decoder, self).__init__()
        self.target_projection = Linear(input_dim-y_dim, num_hidden)
        self.linears = nn.ModuleList([Linear(num_hidden * 3, num_hidden * 3, w_init='relu') for _ in range(3)])
        self.final_projection_mu = Linear(num_hidden * 3, y_dim)
        self.final_projection_std = Linear(num_hidden * 3, y_dim)
        
    def forward(self, r, z, target_x):
        batch_size, num_targets, _ = target_x.size()
        # project vector with dimension 2 --> num_hidden
        target_x = self.target_projection(target_x)
        
        # concat all vectors (r,z,target_x)
        hidden = t.cat([t.cat([r,z], dim=-1), target_x], dim=-1)
        
        # mlp layers
        for linear in self.linears:
            hidden = t.relu(linear(hidden))
            
        # get mu and sigma
        y_pred = self.final_projection_mu(hidden)
        log_sigma = self.final_projection_std(hidden)
        # Bound the variance
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        dist = self.MultivariateNormalDiag(y_pred, sigma)
        
        return y_pred, sigma, dist
    
    def MultivariateNormalDiag(self, loc, scale_diag):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        return Independent(Normal(loc, scale_diag), 1)

class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """
    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden 
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
#        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):
        # Get attention score
        attn = t.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)
        
        attn = t.softmax(attn, dim=-1)

        # Dropout
#        attn = self.attn_dropout(attn)
        
        # Get Context Vector
        result = t.bmm(attn, value)

        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """
    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads 
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

#        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query):

        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query
        
        # Make multihead
        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)
        
        # Concatenate context vector with input (most important)
        result = t.cat([residual, result], dim=-1)
        
        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
#        result = self.residual_dropout(result)
        result = result + residual

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns
    
