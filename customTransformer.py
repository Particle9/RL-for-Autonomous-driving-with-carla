import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, in_dim, num_heads, ff_dim, out_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(in_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, in_dim)
        )
        self.layernorm1 = nn.LayerNorm(in_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(in_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Linear layers for final output
        self.f1 = nn.Linear(1024, 512)
        self.f2 = nn.Linear(512, 256)
        self.f3 = nn.Linear(256, 64)
        self.fout = nn.Linear(64, out_dim)  

    def forward(self, inputs):
        # inputs should have shape (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attention(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        # Pass through the fully connected layers
        x = torch.relu(self.f1(out2))
        x = torch.relu(self.f2(x))
        x = torch.relu(self.f3(x))
        
        # Final output layer
        output = self.fout(x)
        return output  # Output shape: (batch_size, 3) for throttle, brake, steering