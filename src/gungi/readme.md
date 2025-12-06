# Gungi AlphaZero

AlphaZero-style training for the strategic board game Gungi.

[Official Rulebook](https://planetbanatt.net/docs/gungi_rulebook_en.pdf)

## Training Locally

```bash
python src/gungi/train.py \
  --iterations 50 \
  --games-per-iter 10 \
  --mcts-sims 100 \
  --batch-size 64 \
  --epochs 10 \
  --device mps \
  --network-size small
```

### Monitoring Training

Watch live metrics with TensorBoard:

```bash
.venv/bin/tensorboard --logdir runs/gungi
```

Then open `http://localhost:6006` in your browser.

## Training on Modal (Cloud GPU)

For faster training on cloud GPUs (A10G, A100, H100), use Modal with your cloud GPU credits.

### Setup

1. Install Modal CLI:
```bash
pip install modal
modal setup
```

2. Authenticate with your Modal account

### Run Training on GPU

```bash
# Basic training on A10G GPU (~$0.30/hr)
modal run src/gungi/modal.py --iterations 50 --games-per-iter 25

# Faster training on A100 (~$1.10/hr)
modal run src/gungi/modal.py --gpu a100 --iterations 100 --mcts-sims 200

# Fastest training on H100 (~$2.00/hr)
modal run src/gungi/modal.py --gpu h100 --iterations 100 --mcts-sims 400 --workers 8
```

### Resume Training

```bash
modal run src/gungi/modal.py \
  --iterations 100 \
  --resume /root/checkpoints/gungi/gungi_alphazero_iter_50.pt
```

### Download Results

Checkpoints and logs are automatically saved to Modal volumes.

Download them locally:

```bash
# Download checkpoints
modal volume get gungi-checkpoints ./local-checkpoints

# Download TensorBoard logs
modal volume get gungi-logs ./local-logs
```

### Modal Volume Management

List volumes:
```bash
modal volume ls
```

Remove old volumes:
```bash
modal volume rm gungi-checkpoints
modal volume rm gungi-logs
```

## Architecture

### Training Pipeline

1. **Self-Play**: Generate training data with MCTS-guided play
2. **Training**: Update neural network on replay buffer samples
3. **Evaluation**: Test against random player (every 10 iterations)

### Neural Network

- **Input**: 86-plane board state (14 piece types × 2 colors × 3 stack levels + 2 meta planes)
- **Body**: Convolutional layers + residual tower
- **Heads**:
  - Policy head: 8829 action logits
  - Value head: Position evaluation (-1 to +1)

### Network Sizes

- **small**: 5 residual blocks, 64 channels (~1M parameters)
- **standard**: 10 residual blocks, 128 channels (~10M parameters)
- **large**: 20 residual blocks, 256 channels (~50M parameters)

## Learning Goals

- **Setup Phase**: Learn optimal piece placement and stacking strategies
- **Mid-game**: Learn piece movement and positioning tactics
- **End-game**: Learn mating patterns and capture sequences