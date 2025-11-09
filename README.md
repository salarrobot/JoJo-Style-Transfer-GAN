# JoJo Style Transfer GAN

A lightweight CycleGAN implementation for unpaired image-to-image translation from real faces to JoJo's Bizarre Adventure anime style.

## Architecture

### Model Overview

Two generators and two discriminators:
- `G_AB: X → Y` (Real → Anime)
- `G_BA: Y → X` (Anime → Real)  
- `D_A`: Discriminates real anime images from generated
- `D_B`: Discriminates real photos from generated

### Generator Architecture

Simple encoder-decoder structure:

**Encoder:**
```
Input (3×128×128) 
→ Conv2d(3→32, k=7, p=3) + ReLU
→ Conv2d(32→64, k=3, s=2, p=1) + ReLU  
→ Conv2d(64→128, k=3, s=2, p=1) + ReLU
→ Latent (128×32×32)
```

**Decoder:**
```
Latent (128×32×32)
→ ConvTranspose2d(128→64, k=3, s=2, p=1, op=1) + ReLU
→ ConvTranspose2d(64→32, k=3, s=2, p=1, op=1) + ReLU
→ Conv2d(32→3, k=7, p=3) + Tanh
→ Output (3×128×128)
```

Total parameters per generator: **194,051**

### Discriminator Architecture

PatchGAN discriminator outputting 15×15 predictions:

```
Input (3×128×128)
→ Conv2d(3→32, k=4, s=2, p=1) + LeakyReLU(0.2)
→ Conv2d(32→64, k=4, s=2, p=1) + LeakyReLU(0.2)
→ Conv2d(64→128, k=4, s=2, p=1) + LeakyReLU(0.2)  
→ Conv2d(128→1, k=4, p=1)
→ Output (1×15×15)
```

## Mathematical Formulation

### Loss Functions

**1. Adversarial Loss (GAN Loss)**

For generator G_AB and discriminator D_A:

```
L_GAN(G_AB, D_A, X, Y) = E_y~p(y)[log D_A(y)] + E_x~p(x)[log(1 - D_A(G_AB(x)))]
```

We use MSE loss instead of binary cross-entropy:

```
L_GAN(G_AB, D_A) = E_x~p(x)[(D_A(G_AB(x)) - 1)²]
```

**2. Cycle Consistency Loss**

Ensures content preservation through reconstruction:

```
L_cyc(G_AB, G_BA) = E_x~p(x)[||G_BA(G_AB(x)) - x||₁] + E_y~p(y)[||G_AB(G_BA(y)) - y||₁]
```

Forward cycle: Real → Anime → Real  
Backward cycle: Anime → Real → Anime

**3. Identity Loss**

Encourages color/tone consistency:

```
L_identity(G_AB, G_BA) = E_y~p(y)[||G_AB(y) - y||₁] + E_x~p(x)[||G_BA(x) - x||₁]
```

If input is already in target domain, output should be unchanged.

**Total Objective**

```
L_total = L_GAN + λ_cyc × L_cyc + λ_id × L_identity
```

Where:
- λ_cyc = 10 (cycle consistency weight)
- λ_id = 5 (identity loss weight)

The full objective optimized:

```
min_G max_D L_GAN(G_AB, D_A, X, Y) + L_GAN(G_BA, D_B, Y, X) 
          + λ_cyc × L_cyc(G_AB, G_BA)
          + λ_id × L_identity(G_AB, G_BA)
```

## Requirements

```bash
pip install torch torchvision pillow tqdm matplotlib
```

## Dataset Structure

```
├── train/
│   ├── Real_Faces/     # 5000 real face photos  
│   └── Anime_Faces/    # 5000 JoJo anime faces
├── test/               # Test images
├── outputs/samples/    # Training progress samples
├── checkpoints/        # Saved models
└── results/           # Generated outputs
```

## Usage

### Training

Run notebook cells sequentially:

1. **Setup** - Initialize parameters (IMG_SIZE=128, EPOCHS=20, BATCH_SIZE=2)
2. **Load Data** - Load 5000 images from each domain
3. **Build Models** - Create generators, discriminators, optimizers
4. **Train** - Run 20 epochs, save checkpoints every 5 epochs

Training time: ~2-3 hours on GTX 1050

### Inference

```python
# Load model
G_AB.load_state_dict(torch.load('checkpoints/G_AB_final.pt'))
G_AB.eval()

# Transform image
with torch.no_grad():
    anime_style = G_AB(real_image)
```

### Hyperparameters

```python
IMG_SIZE = 128          # Image resolution
EPOCHS = 20            # Training epochs
BATCH_SIZE = 2         # Batch size
LEARNING_RATE = 0.0002 # Adam learning rate
BETA1 = 0.5           # Adam beta1
LAMBDA_CYC = 10       # Cycle consistency weight
LAMBDA_ID = 5         # Identity loss weight
```

## Model Performance

### Computational Efficiency
- **Model Size**: 194K parameters (extremely lightweight!)
- **Memory Usage**: ~2GB GPU RAM during training
- **Inference Speed**: ~50ms per image (RTX 3060)

### Quality Metrics
The model balances three objectives:
1. **Realism**: Generated images should look like authentic JoJo artwork
2. **Identity Preservation**: Facial features and structure should remain recognizable
3. **Style Transfer**: Successfully captures JoJo's distinctive artistic style

## Training Tips

### For Better Results
1. **More epochs**: Train for 50-100 epochs for higher quality
2. **Higher resolution**: Use 256×256 images if GPU memory allows
3. **Data augmentation**: Add random flips and color jitter
4. **Larger dataset**: Use more training images for better generalization

### For Faster Training
1. **Lower resolution**: 64×64 images train 4× faster
2. **Smaller batch size**: Reduces memory usage
3. **Fewer images**: Use subset of dataset for quick experiments

### Common Issues
- **Mode collapse**: If all outputs look similar, reduce learning rate or increase cycle consistency weight
- **Low quality**: Train for more epochs or increase model capacity
- **Color shifts**: Adjust identity loss weight

## File Descriptions

- **`JoJo-Style-Transfer-GAN.ipynb`**: Main training and inference notebook
- **`checkpoints/G_AB_final.pt`**: Trained generator model (Real→Anime)
- **`outputs/samples/`**: Training progress samples (saved every 5 epochs)
- **`results/`**: Generated test results
- **`submission.zip`**: Complete submission package

## Technical Details

### CycleGAN Overview
CycleGAN enables unpaired image-to-image translation by learning two mappings:
- **G_AB**: Real faces → JoJo style
- **G_BA**: JoJo style → Real faces

The cycle consistency constraint ensures that `G_BA(G_AB(image)) ≈ image`, preventing mode collapse and preserving content.

### Why This Approach?
- **No paired data needed**: Train with unpaired images from both domains
- **Preserves identity**: Cycle consistency maintains facial structure
- **Style transfer**: Adversarial training captures artistic style
- **Lightweight**: Efficient architecture suitable for consumer GPUs




For questions or issues, please open an issue on GitHub.

**⭐ Star this repo if you find it useful!**
