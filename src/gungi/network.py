#!/usr/bin/env python3
"""Neural network architecture for AlphaZero-style Gungi agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import NUM_STATE_PLANES, NUM_ACTIONS, BOARD_SIZE


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class PolicyHead(nn.Module):
    """Policy head: outputs logits for all possible actions."""

    def __init__(
        self, num_channels: int, num_actions: int, board_size: int = BOARD_SIZE
    ):
        super().__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * board_size * board_size, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ValueHead(nn.Module):
    """Value head: outputs scalar value estimation [-1, 1]."""

    def __init__(self, num_channels: int, board_size: int = BOARD_SIZE):
        super().__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(board_size * board_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        return out


class GungiNetwork(nn.Module):
    """
    AlphaZero-style neural network for Gungi.

    Architecture:
    - Input: Board state encoding (86 planes of 9x9)
    - Body: Convolutional layer + residual tower
    - Output: Policy head (action probabilities) + Value head (position evaluation)

    Args:
        num_res_blocks: Number of residual blocks (default 10)
        num_channels: Number of channels in residual tower (default 128)
    """

    def __init__(
        self,
        num_res_blocks: int = 10,
        num_channels: int = 128,
        num_input_planes: int = NUM_STATE_PLANES,
        num_actions: int = NUM_ACTIONS,
        board_size: int = BOARD_SIZE,
    ):
        super().__init__()

        self.num_input_planes = num_input_planes
        self.num_actions = num_actions
        self.board_size = board_size

        # Initial convolution
        self.input_conv = nn.Conv2d(
            num_input_planes, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.input_bn = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_head = PolicyHead(num_channels, num_actions, board_size)

        # Value head
        self.value_head = ValueHead(num_channels, board_size)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, num_input_planes, board_size, board_size)

        Returns:
            tuple: (policy_logits, value)
                - policy_logits: Shape (batch, num_actions)
                - value: Shape (batch, 1)
        """
        # Initial convolution
        out = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        for block in self.res_blocks:
            out = block(out)

        # Heads
        policy_logits = self.policy_head(out)
        value = self.value_head(out)

        return policy_logits, value

    def predict(self, state: torch.Tensor, legal_mask: torch.Tensor = None) -> tuple:
        """
        Predict policy and value for a given state.

        Args:
            state: Board state tensor (batch, planes, H, W)
            legal_mask: Optional mask for legal actions (batch, num_actions)

        Returns:
            tuple: (policy_probs, value)
                - policy_probs: Normalized probabilities (batch, num_actions)
                - value: Scalar value estimate (batch, 1)
        """
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(state)

            if legal_mask is not None:
                # Mask illegal actions with large negative value
                policy_logits = policy_logits.masked_fill(
                    legal_mask == 0, float("-inf")
                )

            policy_probs = F.softmax(policy_logits, dim=-1)

            # Handle case where all actions are masked (shouldn't happen in valid game state)
            if legal_mask is not None:
                # Replace NaN with uniform distribution over legal actions
                nan_mask = torch.isnan(policy_probs).any(dim=-1)
                if nan_mask.any():
                    uniform = legal_mask / legal_mask.sum(dim=-1, keepdim=True)
                    policy_probs[nan_mask] = uniform[nan_mask]

        return policy_probs, value

    def save(self, path: str):
        """Save model weights."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "num_res_blocks": len(self.res_blocks),
                "num_channels": self.input_conv.out_channels,
                "num_input_planes": self.num_input_planes,
                "num_actions": self.num_actions,
                "board_size": self.board_size,
            },
            path,
        )
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> "GungiNetwork":
        """Load model from checkpoint."""
        if device is None:
            device = torch.device("cpu")

        checkpoint = torch.load(path, map_location=device)

        model = cls(
            num_res_blocks=checkpoint.get("num_res_blocks", 10),
            num_channels=checkpoint.get("num_channels", 128),
            num_input_planes=checkpoint.get("num_input_planes", NUM_STATE_PLANES),
            num_actions=checkpoint.get("num_actions", NUM_ACTIONS),
            board_size=checkpoint.get("board_size", BOARD_SIZE),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        print(f"Model loaded from {path}")
        return model


class GungiNetworkSmall(GungiNetwork):
    """Smaller network variant for faster training/testing."""

    def __init__(self):
        super().__init__(
            num_res_blocks=5,
            num_channels=64,
        )


class GungiNetworkLarge(GungiNetwork):
    """Larger network variant for stronger play."""

    def __init__(self):
        super().__init__(
            num_res_blocks=20,
            num_channels=256,
        )


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test network creation and forward pass
    import numpy as np

    print("Testing GungiNetwork...")

    # Create network
    net = GungiNetwork(num_res_blocks=5, num_channels=64)
    print(f"Parameters: {count_parameters(net):,}")

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, NUM_STATE_PLANES, BOARD_SIZE, BOARD_SIZE)

    policy_logits, value = net(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value shape: {value.shape}")

    # Test with legal mask
    legal_mask = torch.ones(batch_size, NUM_ACTIONS)
    legal_mask[:, 1000:] = 0  # Mask out some actions

    policy_probs, value = net.predict(dummy_input, legal_mask)
    print(f"Policy probs shape: {policy_probs.shape}")
    print(f"Policy probs sum: {policy_probs.sum(dim=-1)}")  # Should be 1.0

    print("All tests passed!")
