# JoJo Style Transfer GAN 🎨

A lightweight CycleGAN implementation for transforming real face photos into JoJo's Bizarre Adventure anime style. This project uses unpaired image-to-image translation to convert realistic portraits into the distinctive artistic style of the JoJo anime series.

## Results

### Training Progress

<div align="center">

| Epoch 5 | Epoch 10 | Epoch 15 | Epoch 20 |
|---------|----------|----------|----------|
| ![ep5](ep5.png) | ![ep10](ep10.png) | ![ep15](ep15.png) | ![ep20](ep20.png) |

*Top row: Original real photos | Bottom row: Generated JoJo-style images*

</div>

### Sample Transformations

![Preview](preview.png)

The model successfully captures key characteristics of the JoJo anime style:
- Bold line art and stylized features
- Dramatic shading and contrast
- Preservation of facial structure and identity
- Anime-style color palette and textures

## Architecture

This implementation uses a **CycleGAN** architecture optimized for fast training on consumer GPUs:

### Generators
- **Encoder-Decoder architecture** with residual connections
- **Input**: 128×128 RGB images
- **Encoder**: 3 convolutional layers (3→32→64→128 channels)
- **Decoder**: 3 transposed convolutional layers (128→64→32→3 channels)
- **Activation**: ReLU for encoder, Tanh for output
- **Parameters**: ~194K per generator (lightweight!)

### Discriminators
- **PatchGAN discriminator** for realistic texture assessment
- **Architecture**: 4 convolutional layers with LeakyReLU
- **Output**: 15×15 patch predictions

### Loss Functions
- **Adversarial Loss (GAN)**: MSE loss for realistic generation
- **Cycle Consistency Loss**: L1 loss (weight: 10) for preserving content
- **Identity Loss**: L1 loss (weight: 5) for color consistency

**Total Loss**: `L_total = L_GAN + 10×L_cycle + 5×L_identity`

## Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.0.0
tqdm
matplotlib
```

Install dependencies:
```bash
pip install torch torchvision pillow tqdm matplotlib
```

## Dataset Structure

```
project/
├── train/
│   ├── Real_Faces/          # Real face photos (5000+ images)
│   └── Anime_Faces/         # JoJo anime faces (5000+ images)
├── test/                    # Test images for inference
├── outputs/
│   └── samples/            # Training samples saved every 5 epochs
├── checkpoints/            # Model checkpoints
└── results/                # Generated test results
```

### Dataset Notes
- **Training images**: JPEG or PNG format
- **Image size**: Automatically resized to 128×128 during training
- **Quantity**: 5000 images per domain (real and anime)
- **No paired images required** - CycleGAN learns from unpaired data

## Usage

### 1. Training

Open the Jupyter notebook and run cells sequentially:

```python
# Cell 1: Setup and configuration
# - Defines training parameters (IMG_SIZE=128, EPOCHS=20, BATCH_SIZE=2)
# - Creates output directories

# Cell 2: Load dataset
# - Loads 5000 real faces and 5000 anime faces
# - Creates DataLoader

# Cell 3: Initialize models
# - Creates generators (G_AB, G_BA) and discriminators (D_A, D_B)
# - Sets up optimizers and loss functions

# Cell 4: Train
# - Runs training loop for 20 epochs
# - Saves samples every 5 epochs
# - Saves final model to checkpoints/G_AB_final.pt
```

**Training Configuration:**
- **Image Size**: 128×128 pixels
- **Epochs**: 20
- **Batch Size**: 2
- **Learning Rate**: 0.0002 (Adam optimizer)
- **GPU**: Automatically uses CUDA if available

**Expected Training Time**:
- NVIDIA GTX 1050: ~2-3 hours for 20 epochs
- Modern GPUs (RTX 30XX): ~30-60 minutes

### 2. Inference

Generate JoJo-style images from test photos:

```python
# Cell 5: Run inference
# - Loads trained model from checkpoints/
# - Processes all images in test/ folder
# - Saves results to results/ folder
```

### 3. Create Submission Package

```python
# Cell 6: Package results
# - Creates preview visualization
# - Packages model and results into submission.zip
# - Ready for submission/sharing
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

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhu2017unpaired,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  journal={arXiv preprint arXiv:1703.10593},
  year={2017}
}
```

## License

This project is available for educational and research purposes. The JoJo's Bizarre Adventure characters and artwork are © Hirohiko Araki.

## Acknowledgments

- **CycleGAN**: Original paper by Zhu et al. (2017)
- **JoJo's Bizarre Adventure**: Manga and anime by Hirohiko Araki
- **PyTorch**: Deep learning framework
- **Dataset**: Face images from various sources (ensure proper licensing for your use case)

---

## Future Improvements

- [ ] Implement attention mechanisms for better facial feature preservation
- [ ] Add progressive training for higher resolution outputs
- [ ] Experiment with StyleGAN architecture
- [ ] Create web demo for real-time inference
- [ ] Add multi-style support (Part 3, 4, 5 styles)

## Contact

For questions or issues, please open an issue on GitHub.

**⭐ Star this repo if you find it useful!**
