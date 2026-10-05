# video-retalking_legacy

This fork of [OpenTalker/video-retalking](https://github.com/OpenTalker/video-retalking) installs and imports on Python 3.10 with PyTorch 2 and NumPy 2. The models and inference code are the upstream ones. The upstream README follows below, from the VideoReTalking title onward. All changed files are listed in [NOTICE](NOTICE).

## Changes from upstream

- `utils/torchvision_compat.py` (new), imported at the top of `inference.py` and `predict.py`: basicsr 1.4.2 imports `torchvision.transforms.functional_tensor`, which torchvision 0.17 removed. The new module registers `torchvision.transforms.functional` under the old name, so basicsr itself is left unpatched.
- `third_part/face3d/extract_kp_videos.py` and `utils/alignment_stit.py`: use `LandmarksType.TWO_D` when face-alignment has it (1.4 and later) and fall back to `LandmarksType._2D` for 1.3.x.
- `third_part/face3d/util/preprocess.py`: removed the `np.VisibleDeprecationWarning` filter (upstream issue #274).
- `third_part/face_detection/utils.py` and four evaluation scripts under `third_part/face3d/models/arcface_torch/`: `np.int` and `np.float` replaced with `int` and `float`.
- `requirements.txt`: dlib 19.24.6 in place of 19.24.0, numpy 2.2.6 in place of 1.23.4, and scikit-image 0.24.0, tqdm, and `setuptools<81` added. librosa 0.9.2 needs `pkg_resources`, which setuptools 81 removed.
- `train.py` (new, not from upstream): see "About train.py" below.

## Errors this fork fixes

| Error with upstream code and current packages | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'` | torchvision 0.17 or later with basicsr 1.4.2 (upstream issues #224, #246) |
| `AttributeError: _2D` | face-alignment 1.4 or later |
| `AttributeError: module 'numpy' has no attribute 'VisibleDeprecationWarning'` | NumPy 2 (upstream issue #274) |
| `AttributeError: module 'numpy' has no attribute 'int'.` | NumPy 1.24 or later, in `third_part/face_detection/utils.py` |
| `ModuleNotFoundError: No module named 'pkg_resources'` | librosa 0.9.2 with setuptools 81 or later |

## Install

These are the steps used for the test below, with [uv](https://docs.astral.sh/uv/) on macOS:

```
git clone https://github.com/nawta/video-retalking_legacy.git
cd video-retalking_legacy
uv venv -p 3.10
source .venv/bin/activate
uv pip install torch torchvision
uv pip install -r requirements.txt
```

You also need the `ffmpeg` command. On a CUDA machine, pick the `torch` build for your CUDA version from <https://pytorch.org/get-started/locally/>.

On the Apple Silicon Mac used for testing, building dlib 19.24.6 from source failed. With an Intel (Rosetta) `cmake` on the PATH, it built an x86_64 module that Python could not load. With an arm64 `cmake`, the bundled libpng failed with `fatal error: 'fp.h' file not found`. The test below used the prebuilt `dlib-bin` package (20.0.1.post1) instead: `uv pip uninstall dlib && uv pip install dlib-bin`.

The pretrained models are listed in the upstream section below. They total about 3.9 GB and come from <https://github.com/vinthony/video-retalking/releases/tag/v0.0.1>.

## Tested environment

Only installation and imports were tested. Full inference was not run because the checkpoints were not downloaded. CUDA, `webUI.py`, and `predict.py` were not tested.

| Item | Value |
|---|---|
| Machine | Apple M5 Max, macOS 26.5.2, CPU only |
| Python | 3.10.20 (uv) |
| Packages | torch 2.14.1, torchvision 0.29.1, numpy 2.2.6, basicsr 1.4.2, face-alignment 1.3.4, kornia 0.5.1, librosa 0.9.2, scikit-image 0.24.0, opencv-python 5.0.0.93, dlib-bin 20.0.1.post1 |
| `python inference.py --help` | All imports succeed and the help text prints |
| `utils/audio.py` | Computes an (80, 161) mel spectrogram from a 2 s test tone |
| `LandmarksType` fallback | Checked against face-alignment 1.3.4 and 1.5.0 |

## About train.py

`train.py` was added in this fork and does not exist upstream. It is an unfinished outline of the training procedure in the paper and cannot train the models:

- It fails at import with `ModuleNotFoundError: No module named 'face3d'` because it does not add `third_part` to `sys.path`. After adding `third_part` to the path and importing `utils/torchvision_compat.py`, it fails with `AttributeError: 'Namespace' object has no attribute 'DNet_path'` because the argument parser does not define the checkpoint paths that `load_DNet` reads.
- The dataset class returns random tensors, and the lip-sync loss always returns 0.
- The discriminator passes a 64-channel output into a layer that expects 512 channels.

***

<div align="center">

<h2>VideoReTalking <br/> <span style="font-size:12px">Audio-based Lip Synchronization for Talking Head Video Editing in the Wild</span> </h2> 

  <a href='https://arxiv.org/abs/2211.14758'><img src='https://img.shields.io/badge/ArXiv-2211.14758-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://vinthony.github.io/video-retalking/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinthony/video-retalking/blob/main/quick_demo.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![Replicate](https://replicate.com/cjwbw/video-retalking/badge)](https://replicate.com/cjwbw/video-retalking)

<div>
    <a target='_blank'>Kun Cheng <sup>*,1,2</sup> </a>&emsp;
    <a href='https://vinthony.github.io/' target='_blank'>Xiaodong Cun <sup>*,2</a>&emsp;
    <a href='https://yzhang2016.github.io/yongnorriszhang.github.io/' target='_blank'>Yong Zhang <sup>2</sup></a>&emsp;
    <a href='https://menghanxia.github.io/' target='_blank'>Menghan Xia <sup>2</sup></a>&emsp;
    <a href='https://feiiyin.github.io/' target='_blank'>Fei Yin <sup>2,3</sup></a>&emsp;<br/>
    <a href='https://web.xidian.edu.cn/mrzhu/en/index.html' target='_blank'>Mingrui Zhu <sup>1</sup></a>&emsp;
    <a href='https://xuanwangvc.github.io/' target='_blank'>Xuan Wang <sup>2</sup></a>&emsp;
    <a href='https://juewang725.github.io/' target='_blank'>Jue Wang <sup>2</sup></a>&emsp;
    <a href='https://web.xidian.edu.cn/nnwang/en/index.html' target='_blank'>Nannan Wang <sup>1</sup></a>
</div>
<br>
<div>
    <sup>1</sup> Xidian University &emsp; <sup>2</sup> Tencent AI Lab &emsp; <sup>3</sup> Tsinghua University
</div>
<br>
<i><strong><a href='https://sa2022.siggraph.org/' target='_blank'>SIGGRAPH Asia 2022 Conference Track</a></strong></i>
<br>
<br>
<img src="https://opentalker.github.io/video-retalking/static/images/teaser.png" width="768px">


<div align="justify">  <BR> We present VideoReTalking, a new system to edit the faces of a real-world talking head video according to input audio, producing a high-quality and lip-syncing output video even with a different emotion. Our system disentangles this objective into three sequential tasks:
  
 <BR> (1) face video generation with a canonical expression
<BR> (2) audio-driven lip-sync and 
  <BR> (3) face enhancement for improving photo-realism. 
  
 <BR>  Given a talking-head video, we first modify the expression of each frame according to the same expression template using the expression editing network, resulting in a video with the canonical expression. This video, together with the given audio, is then fed into the lip-sync network to generate a lip-syncing video. Finally, we improve the photo-realism of the synthesized faces through an identity-aware face enhancement network and post-processing. We use learning-based approaches for all three steps and all our modules can be tackled in a sequential pipeline without any user intervention.</div>
<BR>

<p>
<img alt='pipeline' src="./docs/static/images/pipeline.png?raw=true" width="768px"><br>
<em align='center'>Pipeline</em>
</p>

</div>

## Results in the Wild （contains audio）
https://user-images.githubusercontent.com/4397546/224310754-665eb2dd-aadc-47dc-b1f9-2029a937b20a.mp4




## Environment
```
# Upstream instructions. For this fork, see "Install" at the top of this file.
git clone https://github.com/vinthony/video-retalking.git
cd video-retalking
conda create -n video_retalking python=3.8
conda activate video_retalking

conda install ffmpeg

# Please follow the instructions from https://pytorch.org/get-started/previous-versions/
# This installation command only works on CUDA 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

## Quick Inference

#### Pretrained Models
Please download our [pre-trained models](https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0?usp=share_link) and put them in `./checkpoints`.

<!-- We also provide some [example videos and audio](https://drive.google.com/drive/folders/14OwbNGDCAMPPdY-l_xO1axpUjkPxI9Dv?usp=share_link). Please put them in `./examples`. -->

#### Inference

```
python3 inference.py \
  --face examples/face/1.mp4 \
  --audio examples/audio/1.wav \
  --outfile results/1_1.mp4
```
This script includes data preprocessing steps. You can test any talking face videos without manual alignment. But it is worth noting that DNet cannot handle extreme poses.

You can also control the expression by adding the following parameters:

```--exp_img```: Pre-defined expression template. The default is "neutral". You can choose "smile" or an image path.

```--up_face```: You can choose "surprise" or "angry" to modify the expression of upper face with [GANimation](https://github.com/donydchen/ganimation_replicate).



## Citation

If you find our work useful in your research, please consider citing:

```
@misc{cheng2022videoretalking,
        title={VideoReTalking: Audio-based Lip Synchronization for Talking Head Video Editing In the Wild}, 
        author={Kun Cheng and Xiaodong Cun and Yong Zhang and Menghan Xia and Fei Yin and Mingrui Zhu and Xuan Wang and Jue Wang and Nannan Wang},
        year={2022},
        eprint={2211.14758},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
  }
```

## Acknowledgement
Thanks to
[Wav2Lip](https://github.com/Rudrabha/Wav2Lip),
[PIRenderer](https://github.com/RenYurui/PIRender), 
[GFP-GAN](https://github.com/TencentARC/GFPGAN), 
[GPEN](https://github.com/yangxy/GPEN),
[ganimation_replicate](https://github.com/donydchen/ganimation_replicate),
[STIT](https://github.com/rotemtzaban/STIT)
for sharing their code.


## Related Work
- [StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pre-trained StyleGAN (ECCV 2022)](https://github.com/FeiiYin/StyleHEAT)
- [CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior (CVPR 2023)](https://github.com/Doubiiu/CodeTalker)
- [SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023)](https://github.com/Winfredy/SadTalker)
- [DPE: Disentanglement of Pose and Expression for General Video Portrait Editing (CVPR 2023)](https://github.com/Carlyx/DPE)
- [3D GAN Inversion with Facial Symmetry Prior (CVPR 2023)](https://github.com/FeiiYin/SPI/)
- [T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations (CVPR 2023)](https://github.com/Mael-zys/T2M-GPT)

##  Disclaimer

This is not an official product of Tencent. 

```
1. Please carefully read and comply with the open-source license applicable to this code before using it. 
2. Please carefully read and comply with the intellectual property declaration applicable to this code before using it.
3. This open-source code runs completely offline and does not collect any personal information or other data. If you use this code to provide services to end-users and collect related data, please take necessary compliance measures according to applicable laws and regulations (such as publishing privacy policies, adopting necessary data security strategies, etc.). If the collected data involves personal information, user consent must be obtained (if applicable). Any legal liabilities arising from this are unrelated to Tencent.
4. Without Tencent's written permission, you are not authorized to use the names or logos legally owned by Tencent, such as "Tencent." Otherwise, you may be liable for your legal responsibilities.
5. This open-source code does not have the ability to directly provide services to end-users. If you need to use this code for further model training or demos, as part of your product to provide services to end-users, or for similar use, please comply with applicable laws and regulations for your product or service. Any legal liabilities arising from this are unrelated to Tencent.
6. It is prohibited to use this open-source code for activities that harm the legitimate rights and interests of others (including but not limited to fraud, deception, infringement of others' portrait rights, reputation rights, etc.), or other behaviors that violate applicable laws and regulations or go against social ethics and good customs (including providing incorrect or false information, spreading pornographic, terrorist, and violent information, etc.). Otherwise, you may be liable for your legal responsibilities.

```
## All Thanks To Our Contributors 

<a href="https://github.com/OpenTalker/video-retalking/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OpenTalker/video-retalking" />
</a>
