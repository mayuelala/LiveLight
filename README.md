# LiveLight

Official implementation of **LiveLight: Real-time Streaming Video Relighting with Interactive Control**.

LiveLight is a real-time streaming video relighting framework with interactive control over light position, intensity, and color.

## Setup Environment

Clone the repository and create a Python environment:

```bash
git clone https://github.com/JiangmingWang1029/LiveLight.git
cd LiveLight

conda create -n livelight python=3.10 -y
conda activate livelight

python -m pip install --upgrade pip
pip install -r requirements.txt
accelerate config
```

### Download LiveLight Weights

Install ModelScope and download the released LiveLight weights:

```bash
python -m pip install modelscope
python download_weights.py
```

The script runs the equivalent ModelScope command:

```bash
modelscope download --model wjm1029/LiveLight \
  --local_dir pretrained_weights/LiveLight
```

By default, the weights are downloaded from `wjm1029/LiveLight` to `pretrained_weights/LiveLight`. A different destination can be selected with:

```bash
python download_weights.py --output-dir path/to/LiveLight_weights
```

The ModelScope repository provides our trained denoising UNet, reference UNet, temporal module, and light guider weights. Update the corresponding paths in `configs/train/` and `configs/prompts/` after downloading them.

The Stable Diffusion Image Variations base model, VAE, image encoder, and other third-party weights are not included. Download them separately from Hugging Face. The default configurations expect a directory structure similar to:

```text
pretrained_weights/
├── sd-image-variations-diffusers/
├── sd-vae-ft-mse/
├── pixel-perfect-depth/
└── xnemo/
```

## Training

Update the dataset paths, pretrained model paths, output directory, and checkpoint paths in the configuration files before launching training.

### Stage 1

```bash
accelerate launch train_livelight_stage1.py \
  --config configs/train/relight_stage1.yaml
```

### Stage 2

Set `warm_start_dir` in `configs/train/relight_stage2.yaml` to the Stage 1 checkpoint directory, then run:

```bash
accelerate launch train_livelight_stage2.py \
  --config configs/train/relight_stage2.yaml
```

### Stage 3

Set the Stage 2 checkpoint paths and temporal module path in `configs/train/relight_stage3_finetune.yaml`, then run:

```bash
accelerate launch train_livelight_stage3_perframe_ref.py \
  --config configs/train/relight_stage3_finetune.yaml
```

## Inference

### Stage 1 Image Inference

```bash
python inference_livelight_stage1.py \
  --input-image path/to/input.png \
  --depth-npy path/to/input_depth.npy \
  --output-dir outputs/stage1 \
  --ckpt-dir path/to/stage1_checkpoint_dir \
  --train-config configs/train/relight_stage1.yaml \
  --use-xformers
```

The light can be controlled with `--light-u`, `--light-v`, `--light-z-rel`, `--light-intensity`, and `--light-color`.

### Stage 3 Video Inference

First update the model, temporal module, and depth estimator paths in `configs/prompts/relight_perframe_ref.yaml`. Prepare the input video as an ordered directory of image frames, then run:

```bash
python inference_livelight_stage3.py \
  --config-path configs/prompts/relight_perframe_ref.yaml \
  --input-dir path/to/input_frames \
  --depth-dir path/to/depth_maps \
  --output-dir outputs/stage3 \
  --num-frames 40 \
  --acceleration xformers
```

`--depth-dir` is optional when the Pixel Perfect Depth paths in the prompt configuration are correctly configured. Predicted frames are saved under `<output-dir>/frames`, together with inference metadata and reports.

## License

See [LICENSE](LICENSE) for details.
