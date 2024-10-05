import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, in_dim, num_heads, ff_dim, out_dim, num_encoder_layers=6, num_decoder_layers=6, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout_rate)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=in_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout_rate)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Layer Normalization
        self.layernorm1 = nn.LayerNorm(in_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(in_dim, eps=1e-6)

        # Fully connected layers for the final output (Throttle, Brake, Steering)
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc_out = nn.Linear(64, out_dim)

    def forward(self, src, tgt):
        # src and tgt should have shape (seq_len, batch_size, embed_dim)
        
        # Encoder
        encoded_src = self.encoder(src)  # Pass source input through the encoder
        
        # Decoder (uses the encoded source and the target)
        decoded_output = self.decoder(tgt, encoded_src)  # Pass target and encoded source through decoder
        
        # Layer normalization after encoder and decoder
        norm_output = self.layernorm1(decoded_output)
        
        # Fully connected layers for final task-specific output
        x = torch.relu(self.fc1(norm_output))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        # Final output layer (e.g., throttle, brake, steering predictions)
        output = self.fc_out(x)
        
        return output