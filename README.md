# Sprite Upscaler - AI-Powered Image Upscaling with Transparency Handling

This repository provides a comprehensive pipeline for upscaling images while preserving transparency using multiple AI models. It includes testing codes, pretrained models, a network interpolation demo and advanced metrics visualization.

## Key Features
- Dual background technique for accurate transparency preservation
Support for various state-of-the-art AI upscaling models
Comprehensive image quality metrics (MSE, PSNR, SSIM)
Flexible output resizing options with scale or dimension control
Model comparison and benchmarking with automated metrics
Batch processing with progress tracking
JSON metrics export for detailed analysis

## Quick Start
#### Dependencies
- Python 3
- [PyTorch >= 1.0](https://pytorch.org/) (CUDA version >= 7.5 if installing with CUDA. [More details](https://pytorch.org/get-started/previous-versions/))
- Required packages: `pip install numpy opencv-python pillow torch spandrel matplotlib pandas`

### Test models
1. Clone this github repo.
```
git clone https://github.com/Nedas127/Sprite-upscaler
cd Sprite-upscaler
```
2. Place your own **low-resolution images** in `./LR` folder. (There are fifteen sample images - various „OnlyFortress" 2D mobile game sprites).
3. Download pretrained models from [Google Drive](https://drive.google.com/drive/folders/1fds6bTbmZJxGoW8pteWWR9sfNTmZcX2N?usp=drive_link). Place the models in `./models`. I provide twenty six different models that varies in architecture and category type. Most of the pre-trained models can be found in https://openmodeldb.info/
4. Run test. I provide models like (4x_PixelPerfectV4_137000_G, 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN, RRDB_ESRGAN_x4, RRDB_PSNR and 22 other). You can configure which model you want to load in the `test.py`. If you want to load them all specify "none"
```
python test.py
```
5. The results are in `./results` folder.
### Network interpolation demo
You can interpolate the RRDB_ESRGAN and RRDB_PSNR or any other models with alpha in [0, 1].

1. Run `python net_interp.py 0.8`, where *0.8* is the interpolation parameter and you can change it to any value in [0,1].
2. Run `python test.py models/interp_08.pth`, where *models/interp_08.pth* is the model path.

<p align="center">
  <img height="400" src="figures/43074.gif">
</p>

### Metrics Visualization
After running the tests, you can generate detailed visualization charts for model performance analysis:

Configure the visualization script: Update the model name and paths in the visualization script:

1. `pythonmodel_name = "your_model_name"`  # e.g., "4x_foolhardy_Remacri"
2. `base_dir = r"path_to_your_results"`  # e.g., r"C:\Users\username\ESRGAN\all_models_results"

Generated outputs:

Interactive matplotlib charts showing PSNR, SSIM, and MSE metrics
High-resolution PNG diagram saved as metrics_diagram.png
Professional styling with color-coded metrics and grid overlays

### Pipeline Architecture
Core Components:

ImageProcessor: Handles transparency extraction and image resizing
ModelProcessor: Manages model loading and upscaling operations
MetricsCalculator: Computes MSE, PSNR, and SSIM quality metrics
UpscalingPipeline: Orchestrates the complete processing workflow
MetricsVisualizer: Generates professional charts and statistical analysis

Transparency Preservation Algorithm
The dual background technique works by:

Dual Processing: Upscale the image on both white and black backgrounds
Alpha Extraction: Calculate transparency from the difference between results
Color Recovery: Derive true RGB values using alpha blending mathematics
Post-Processing: Apply edge refinement and noise reduction (optional)

Quality Metrics
Each processed image is evaluated using:

MSE (Mean Squared Error): Lower is better (displayed with 3 decimal precision)
PSNR (Peak Signal-to-Noise Ratio): Higher is better (measured in dB, 2 decimal precision)
SSIM (Structural Similarity Index): Higher is better (0-1 scale, 2 decimal precision)

### Enhanced Super-Resolution Generative Adversarial Networks
This pipeline supports various ESRGAN-based models and other state-of-the-art upscaling architectures (etc. SwinIR):

ESRGAN variants: RRDB_ESRGAN_x4, RRDB_PSNR
Specialized models: 4x_PixelPerfectV4_137000_G, 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN
And 22+ additional pretrained models

Based on the Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN) research by Xintao Wang et al., which won first place in PIRM2018-SR competition.

By Xintao Wang, [Ke Yu](https://yuke93.github.io/), Shixiang Wu, [Jinjin Gu](http://www.jasongt.com/), Yihao Liu, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=en), [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/), [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/)

The paper is accepted to [ECCV2018 PIRM Workshop](https://pirm2018.org/).

#### BibTeX

    @InProceedings{wang2018esrgan,
        author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
        title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
        booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
        month = {September},
        year = {2018}
    }

<p align="center">
  <img src="figures/baboon.jpg">
</p>

The **RRDB_PSNR** PSNR_oriented model trained with DF2K dataset (a merged dataset with [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (proposed in [EDSR](https://github.com/LimBee/NTIRE2017))) is also able to achive high PSNR performance.

| <sub>Method</sub> | <sub>Training dataset</sub> | <sub>Set5</sub> | <sub>Set14</sub> | <sub>BSD100</sub> | <sub>Urban100</sub> | <sub>Manga109</sub> |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| <sub>[SRCNN](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)</sub>| <sub>291</sub>| <sub>30.48/0.8628</sub> |<sub>27.50/0.7513</sub>|<sub>26.90/0.7101</sub>|<sub>24.52/0.7221</sub>|<sub>27.58/0.8555</sub>|
| <sub>[EDSR](https://github.com/thstkdgus35/EDSR-PyTorch)</sub> | <sub>DIV2K</sub> | <sub>32.46/0.8968</sub> | <sub>28.80/0.7876</sub> | <sub>27.71/0.7420</sub> | <sub>26.64/0.8033</sub> | <sub>31.02/0.9148</sub> |
| <sub>[RCAN](https://github.com/yulunzhang/RCAN)</sub> |  <sub>DIV2K</sub> | <sub>32.63/0.9002</sub> | <sub>28.87/0.7889</sub> | <sub>27.77/0.7436</sub> | <sub>26.82/ 0.8087</sub>| <sub>31.22/ 0.9173</sub>|
|<sub>RRDB(ours)</sub>| <sub>DF2K</sub>| <sub>**32.73/0.9011**</sub> |<sub>**28.99/0.7917**</sub> |<sub>**27.85/0.7455**</sub> |<sub>**27.03/0.8153**</sub> |<sub>**31.66/0.9196**</sub>|

## Perceptual-driven SR Results

You can download ESRGAN research results from [Google Drive](https://drive.google.com/drive/folders/1iaM-c6EgT1FNoJAOKmDrK7YhEhtlKcLx?usp=sharing). (:heavy_check_mark: included;  :heavy_minus_sign: not included; :o: TODO)

HR images can be downloaed from [BasicSR-Datasets](https://github.com/xinntao/BasicSR#datasets).

| Datasets |LR | [*ESRGAN*](https://arxiv.org/abs/1809.00219) | [SRGAN](https://arxiv.org/abs/1609.04802) | [EnhanceNet](http://openaccess.thecvf.com/content_ICCV_2017/papers/Sajjadi_EnhanceNet_Single_Image_ICCV_2017_paper.pdf) | [CX](https://arxiv.org/abs/1803.04626) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Set5 |:heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:| :o: |
| Set14 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:| :o: |
| BSDS100 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:| :o: |
| [PIRM](https://pirm.github.io/) <br><sup>(val, test)</sup> | :heavy_check_mark: | :heavy_check_mark: | :heavy_minus_sign: | :heavy_check_mark:| :heavy_check_mark: |
| [OST300](https://arxiv.org/pdf/1804.02815.pdf) |:heavy_check_mark: | :heavy_check_mark: | :heavy_minus_sign: | :heavy_check_mark:| :o: |
| urban100 | :heavy_check_mark: | :heavy_check_mark: | :heavy_minus_sign: | :heavy_check_mark:| :o: |
| [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) <br><sup>(val, test)</sup> | :heavy_check_mark: | :heavy_check_mark: | :heavy_minus_sign: | :heavy_check_mark:| :o: |

## ESRGAN
We improve the [SRGAN](https://arxiv.org/abs/1609.04802) from three aspects:
1. adopt a deeper model using Residual-in-Residual Dense Block (RRDB) without batch normalization layers.
2. employ [Relativistic average GAN](https://ajolicoeur.wordpress.com/relativisticgan/) instead of the vanilla GAN.
3. improve the perceptual loss by using the features before activation.

In contrast to SRGAN, which claimed that **deeper models are increasingly difficult to train**, our deeper ESRGAN model shows its superior performance with easy training.

<p align="center">
  <img height="120" src="figures/architecture.jpg">
</p>
<p align="center">
  <img height="180" src="figures/RRDB.png">
</p>

## Network Interpolation
We propose the **network interpolation strategy** to balance the visual quality and PSNR.

<p align="center">
  <img height="500" src="figures/net_interp.jpg">
</p>

We show the smooth animation with the interpolation parameters changing from 0 to 1.
Interestingly, it is observed that the network interpolation strategy provides a smooth control of the RRDB_PSNR model and the fine-tuned ESRGAN model.

<p align="center">
  <img height="480" src="figures/81.gif">
  &nbsp &nbsp
  <img height="480" src="figures/102061.gif">
</p>

## Qualitative Results
PSNR (evaluated on the Y channel) and the perceptual index used in the PIRM-SR challenge are also provided for reference.

<p align="center">
  <img src="figures/qualitative_cmp_01.jpg">
</p>
<p align="center">
  <img src="figures/qualitative_cmp_02.jpg">
</p>
<p align="center">
  <img src="figures/qualitative_cmp_03.jpg">
</p>
<p align="center">
  <img src="figures/qualitative_cmp_04.jpg">
</p>

## Ablation Study
Overall visual comparisons for showing the effects of each component in
ESRGAN. Each column represents a model with its configurations in the top.
The red sign indicates the main improvement compared with the previous model.
<p align="center">
  <img src="figures/abalation_study.png">
</p>

## BN artifacts
We empirically observe that BN layers tend to bring artifacts. These artifacts,
namely BN artifacts, occasionally appear among iterations and different settings,
violating the needs for a stable performance over training. We find that
the network depth, BN position, training dataset and training loss
have impact on the occurrence of BN artifacts.
<p align="center">
  <img src="figures/BN_artifacts.jpg">
</p>

## Performance Notes
The pipeline automatically uses CUDA if available

Processing time varies by model complexity and image size

Memory requirements depend on model size and input dimensions

## Credits
This project builds upon:

ESRGAN by Xintao Wang et al.

BasicSR framework

Spandrel https://github.com/chaiNNer-org/spandrel

Various pre-trained models from OpenModelDB
