import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import rmsnorm , apply_rope

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim=1024, q_dim=2048, num_heads_q=16, num_heads_kv=8, mlp_dim=3072):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_dim = q_dim
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim_q = q_dim // num_heads_q
        self.head_dim_kv = hidden_dim // num_heads_kv

        self.q_proj = nn.Linear(hidden_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, hidden_dim, bias=False)

        self.q_norm = nn.Parameter(torch.ones(self.head_dim_q))
        self.k_norm = nn.Parameter(torch.ones(self.head_dim_kv))

        self.input_layernorm = nn.Parameter(torch.ones(hidden_dim))
        self.post_attention_layernorm = nn.Parameter(torch.ones(hidden_dim))

        self.gate_proj = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.up_proj   = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, hidden_dim, bias=False)
        self.o_proj.is_residual = True
        self.down_proj.is_residual = True

    def forward(self, x):
        if x.dim() == 2:  # [T, hidden_dim] -> add batch dim
            x = x.unsqueeze(0)
        B, T, _ = x.shape  # [B, T, hidden_dim]

        x_norm = rmsnorm(x, self.input_layernorm)  # [B, T, hidden_dim]

        Q = self.q_proj(x_norm)  # [B, T, q_dim]
        K = self.k_proj(x_norm)  # [B, T, hidden_dim]
        V = self.v_proj(x_norm)  # [B, T, hidden_dim]

        # ---- split into heads ----
        Q = Q.view(B, T, self.num_heads_q, self.head_dim_q).transpose(1, 2)  # [B, H_q, T, D]
        K = K.view(B, T, self.num_heads_kv, self.head_dim_kv).transpose(1, 2)  # [B, H_kv, T, D]
        V = V.view(B, T, self.num_heads_kv, self.head_dim_kv).transpose(1, 2)  # [B, H_kv, T, D]

        Q = rmsnorm(Q, self.q_norm)  # [B, H_q, T, D]
        K = rmsnorm(K, self.k_norm)  # [B, H_kv, T, D]

        # ---- repeat K/V to match Q heads (MQA) ----
        num_groups = self.num_heads_q // self.num_heads_kv
        K = K.repeat_interleave(num_groups, dim=1)  # [B, H_q, T, D]
        V = V.repeat_interleave(num_groups, dim=1)  # [B, H_q, T, D]
        
        # ---- apply RoPE ----
        Q, K = apply_rope(Q, K, seq_len=T)  # [B, H_q, T, D]

        # ---- scaled dot-product attention ----
        attn_scores = torch.einsum('bhid,bhjd->bhij', Q, K)  # [B, H_q, T, T]
        attn_scores = attn_scores / (self.head_dim_q ** 0.5)

        # ---- causal mask ----
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)  # [B, H_q, T, T]

        # ---- attention output ----
        attn_out = torch.einsum('bhij,bhjd->bhid', attn_probs, V)  # [B, H_q, T, D]
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.q_dim)  # [B, T, q_dim]

        # ---- output projection + residual ----
        x_attn = x + self.o_proj(attn_out)  # [B, T, hidden_dim]
        x_mlp_norm = rmsnorm(x_attn, self.post_attention_layernorm)  # [B, T, hidden_dim]

        # ---- MLP with GLU ----
        gate = self.gate_proj(x_mlp_norm)  # [B, T, mlp_dim]
        up   = self.up_proj(x_mlp_norm)    # [B, T, mlp_dim]
        hidden = F.silu(gate) * up         # [B, T, mlp_dim]
        mlp_out = self.down_proj(hidden)   # [B, T, hidden_dim]

        # ---- final residual ----
        x_out = x_attn + mlp_out           # [B, T, hidden_dim]

        if x_out.shape[0] == 1 and x.dim() == 2:
            x_out = x_out.squeeze(0)  # remove batch dim if original input had none

        return x_out  # [B, T, hidden_dim] or [T, hidden_dim] if single sequence



class TransformerModel(nn.Module):
    def __init__(self, 
                 vocab_size=151936, 
                 hidden_dim=1024, 
                 q_dim=2048, 
                 num_heads_q=16, 
                 num_heads_kv=8, 
                 mlp_dim=3072, 
                 num_layers=28,
                 tie_embeddings=False): 
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim=hidden_dim,
                             q_dim=q_dim,
                             num_heads_q=num_heads_q,
                             num_heads_kv=num_heads_kv,
                             mlp_dim=mlp_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.Parameter(torch.ones(hidden_dim))
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
          
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        scaled_std = std / (2 * self.num_layers) ** 0.5

        if isinstance(module, nn.Linear):
            if hasattr(module, 'is_residual') and module.is_residual:
                nn.init.normal_(module.weight, mean=0.0, std=scaled_std)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, token_ids, targets=None):
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        B, T = token_ids.shape
        x = self.embed_tokens(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = rmsnorm(x, self.norm)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if targets.dim() == 1:
                targets = targets.unsqueeze(0)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        if squeeze_batch:
            logits = logits.squeeze(0)
            return (logits, loss) if loss is not None else logits

        return (logits, loss) if loss is not None else logits
