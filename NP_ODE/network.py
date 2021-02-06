from module import *

class LatentModel(nn.Module):
    """
    Latent Model (Attentive Neural Process)
    """
    def __init__(self, num_hidden, input_dim, y_dim):
        super(LatentModel, self).__init__()
        self.latent_encoder = LatentEncoder(num_hidden, num_hidden, input_dim)
        self.deterministic_encoder = DeterministicEncoder(num_hidden, num_hidden, input_dim, y_dim)
        self.decoder = Decoder(num_hidden, input_dim, y_dim)
        self.BCELoss = nn.BCELoss()
        self.L1loss = nn.L1Loss()
        self.MSEloss = nn.MSELoss()
        self.kl = nn.KLDivLoss()
        
    def forward(self, context_x, context_y, target_x, target_y=None):
        num_targets = target_x.size(1)
        
        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)
        
        # For training
        if target_y is not None:
            posterior_mu, posterior_var, posterior = self.latent_encoder(target_x, target_y)
            z = posterior
        
        # For Generation
        else:
            z = prior
        
        z = z.unsqueeze(1).repeat(1,num_targets,1) # [B, T_target, H]
        r = self.deterministic_encoder(context_x, context_y, target_x) # [B, T_target, H]
        
        # mu should be the prediction of target y
        y_pred, sigma, dist = self.decoder(r, z, target_x)
        
        # For Training
        if target_y is not None:
            # get log probability
            log_likelihood = dist.log_prob(target_y).mean()
            
#            bce_loss = self.BCELoss(t.sigmoid(y_pred), target_y)
            
            # get KL divergence between prior and posterior
            kl = self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var)
#            kl = self.kl(prior_mu, posterior_mu)
            l1 = (t.abs(y_pred - target_y) / target_y).mean()
#            self.L1loss(y_pred, target_y) 
            MSE = self.MSEloss(y_pred, target_y)
            
            # maximize prob and minimize KL divergence
#            loss = -log_likelihood + kl + l1
            loss = -log_likelihood + kl
        
        # For Generation
        else:
            log_p = None
            kl = None
            loss = None
        
        return y_pred, sigma, kl, loss
    
    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (t.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / t.exp(prior_var) - 1. + (prior_var - posterior_var)
#        kl_div = 0.5 * kl_div.sum()
        kl_div = kl_div.mean()
        return kl_div