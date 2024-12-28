import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from dataclasses import dataclass
from typing import Union, Optional, List, Tuple
from diffusers import DiffusionPipeline
import torch
from torch import nn, einsum
import torch.nn.functional as F

# Custom implementation of the GELU activation function for improved model performance
def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# Self-Attention mechanism as used in Transformer-based architectures
class SelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by the number of heads"
        
        # Linear layers for computing queries, keys, and values for attention
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)  # Output projection
        self.attn_dropout = nn.Dropout(dropout)  # Dropout for attention probabilities
        self.resid_dropout = nn.Dropout(dropout) # Dropout for residual connection
        
        self.n_head = n_head  # Number of attention heads
        self.n_embd = n_embd  # Embedding dimensionality

    def forward(self, x):
        B, T, C = x.size()  # B: Batch size, T: Sequence length, C: Embedding dimension
        
        # Compute queries (q), keys (k), and values (v)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape and transpose for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled dot-product attention with softmax normalization
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)  # Apply dropout to attention scores
        
        # Weighted sum of values (output of attention mechanism)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Reassemble head outputs
        
        # Apply output projection and residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y

# Feed-forward neural network layer (used within Transformer blocks)
class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        # Two linear transformations with a GELU activation in between
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

    def forward(self, x):
        x = self.c_fc(x)  # First linear transformation
        x = new_gelu(x)   # Apply custom GELU activation
        x = self.c_proj(x)  # Project back to original embedding size
        x = self.dropout(x)  # Apply dropout
        return x

# Transformer block consisting of self-attention and feed-forward layers
class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)  # Layer normalization before self-attention
        self.attn = SelfAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)  # Layer normalization before feed-forward network
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x, t):
        # Add diffusion timestep embedding (optional)
        x = x + self.attn(self.ln_1(x))  # Apply self-attention with residual connection
        x = x + self.mlp(self.ln_2(x))  # Apply feed-forward network with residual connection
        return x

# Sinusoidal positional encoding for embedding temporal information
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, max_timesteps: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_timesteps = float(max_timesteps)

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.max_timesteps) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # Combine sine and cosine embeddings
        return emb

# Positional embedding layer with sinusoidal encoding
class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_timesteps: int):
        super().__init__()
        self.pe = SinusoidalPosEmb(dim=dim, max_timesteps=max_timesteps)

    def forward(self, x):
        return self.pe(x)  # Compute positional encoding

@dataclass
class GPTConfig:
    dim: int  # Input dimension
    block_size: int = 10000  # Maximum sequence length
    n_layer: int = 12  # Number of transformer layers
    n_head: int = 12  # Number of attention heads
    n_embd: int = 128  # Embedding dimension
    dropout: float = 0.1  # Dropout rate for regularization

# Core GPT model class inheriting from nn.Module
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Ensure a valid block size is set
        assert config.block_size is not None
        self.config = config

        # Define the transformer components
        self.transformer = nn.ModuleDict({
            "input_proj": nn.Linear(config.dim, config.n_embd, bias=False),  # Project input to embedding space
            "wte": nn.Embedding(config.block_size, config.n_embd),  # Token embeddings
            "wpe": PositionalEmbedding(config.n_embd, max_timesteps=config.block_size),  # Positional embeddings
            "drop": nn.Dropout(config.dropout),  # Dropout layer
            "h": nn.ModuleList([
                Block(n_embd=config.n_embd, n_head=config.n_head, dropout=config.dropout)
                for _ in range(config.n_layer)
            ]),  # Stack of transformer blocks
            "ln_f": nn.LayerNorm(config.n_embd),  # Final layer normalization
            "output_proj": nn.Linear(config.n_embd, config.dim, bias=False),  # Project back to input dimension
        })

        # Report the number of parameters in the model
        n_params = sum(p.numel() for p in self.parameters())
        embd_params = sum(p.numel() for p in self.transformer.wte.parameters())
        print(f"Total number of parameters: {n_params / 1e6:.2f}M")
        print(f"Non-Embedding number of parameters: {(n_params - embd_params) / 1e6:.2f}M")
        print(f"Embedding number of parameters: {embd_params / 1e6:.2f}M")

    def forward(self, x, time):
        device = x.device
        b, d, t = x.size()  # [B, C, T]
        assert d == self.config.dim, f"Expected input dim {self.config.dim}, got {d}"

        # Prepare time input and ensure batch size alignment
        time = time.to(dtype=torch.long, device=device)
        if time.ndim == 0:
            time = time.unsqueeze(0).repeat(b)
        assert x.size(0) == time.size(0), "Batch sizes of time and input must match"

        # Transform input through the model
        x_input = x
        x = x.transpose(1, 2)  # [B, T, C]
        inp_emb = self.transformer.input_proj(x)  # Project input to embedding space
        seq_emb = self.transformer.wte(torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0))  # Token embeddings
        time_emb = self.transformer.wpe(time).unsqueeze(1)  # Time embeddings

        # Combine input, sequence, and time embeddings
        x = self.transformer.drop(inp_emb + seq_emb)
        for block in self.transformer.h:
            x = block(x, time_emb)  # Process through transformer blocks
        x = self.transformer.ln_f(x)  # Final normalization
        x_out = self.transformer.output_proj(x)  # Output projection
        x_out = x_out.transpose(1, 2)

        # Residual connection
        x = x_input + x_out
        return x

    # Properties for handling model device and data type
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

# Data structure for pipeline output
@dataclass
class SequenceDataOutput:
    sequences: List[torch.Tensor]
    history: List[torch.Tensor]

# Pipeline for sequence generation using diffusion
class SequenceDDPMPipeline(DiffusionPipeline):
    def __init__(self, model: GPT, scheduler: DDPMScheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self, 
        batch_size: int = 1, 
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, 
        num_inference_steps: int = DIFFUSION_TIMESTEPS, 
        return_dict: bool = True, 
        return_all: bool = False, 
        explicit_noise: torch.Tensor = None, 
        **kwargs,
    ) -> SequenceDataOutput:
        # Initialize the sequence with random noise if not provided
        seq_len = kwargs.get("seq_len", timesteps)
        seq_dim = kwargs.get("channels", channels)
        if explicit_noise is None:
            sequence_data = torch.randn(batch_size, seq_dim, seq_len, device=self.device, generator=generator)
        else:
            sequence_data = explicit_noise.to(device=self.device)

        # Configure the scheduler
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        # Optionally store intermediate results
        history = [sequence_data.cpu()] if return_all else None

        # Reverse diffusion process
        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = self.model(sequence_data, t)  # Predict noise
            sequence_data = self.scheduler.step(model_output, t, sequence_data, generator=generator).prev_sample
            if history is not None:
                history.append(sequence_data.cpu().clone())

        # Clamp output to valid range
        sequence_data = sequence_data.clamp(min=-1.0, max=1.0)
        if not return_dict:
            return (sequence_data,)

        return SequenceDataOutput(sequences=sequence_data, history=history)
