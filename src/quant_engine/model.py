"""
Transformer-based model for price action prediction.

Implements a Transformer encoder architecture for binary classification
of buy/hold signals based on technical indicators.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.

    Adds positional information to input embeddings using sine/cosine functions.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PriceActionTransformer(nn.Module):
    """
    Transformer encoder model for price action prediction.

    Architecture:
        Input -> Linear Projection -> Positional Encoding
             -> Transformer Encoder (2 layers, 4 heads)
             -> Global Average Pooling -> Linear Head -> Sigmoid
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        use_sigmoid: bool = True
    ):
        """
        Initialize the Transformer model.

        Args:
            input_dim: Number of input features (technical indicators)
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of Transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.use_sigmoid = use_sigmoid

        # Input projection layer
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # Input shape: (batch, seq, feature)
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output head
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
            With sigmoid if use_sigmoid=True, logits otherwise
        """
        # Input projection: (batch, seq, input_dim) -> (batch, seq, d_model)
        x = self.input_projection(x)

        # Positional encoding
        # Note: pos_encoder expects (seq, batch, d_model) for buffer access
        # but we use batch_first=True in TransformerEncoder
        x = x.permute(1, 0, 2)  # (seq, batch, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # Back to (batch, seq, d_model)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq, d_model)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)

        # Output head
        x = self.fc_out(x)  # (batch, 1)

        # Sigmoid activation for binary classification (optional)
        if self.use_sigmoid:
            x = torch.sigmoid(x)

        return x

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMBaseline(nn.Module):
    """
    LSTM baseline model for comparison.

    Simple bidirectional LSTM for binary classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize LSTM model.

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take last hidden state
        # h_n shape: (num_layers * num_directions, batch, hidden_dim)
        # We want the last layer's hidden state from both directions
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)

        # Output head
        out = self.fc(hidden)

        return out


if __name__ == "__main__":
    # Test model initialization and forward pass
    batch_size = 32
    seq_len = 64
    input_dim = 14  # Number of technical indicators

    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim)

    # Test Transformer model
    print("Testing PriceActionTransformer:")
    model = PriceActionTransformer(input_dim=input_dim)
    print(f"  Parameters: {model.get_num_params():,}")

    output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Test LSTM baseline
    print("\nTesting LSTMBaseline:")
    lstm_model = LSTMBaseline(input_dim=input_dim)
    print(f"  Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")

    output_lstm = lstm_model(x)
    print(f"  Output shape: {output_lstm.shape}")
    print(f"  Output range: [{output_lstm.min().item():.4f}, {output_lstm.max().item():.4f}]")

    # Test GPU compatibility
    if torch.cuda.is_available():
        print("\nTesting CUDA compatibility:")
        device = torch.device("cuda")
        model_gpu = model.to(device)
        x_gpu = x.to(device)

        output_gpu = model_gpu(x_gpu)
        print(f"  GPU forward pass successful")
        print(f"  Output device: {output_gpu.device}")
