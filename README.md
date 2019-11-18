# Noise2Noise and ImageDetection

This is an unofficial and partial Keras implementation of "Noise2Noise: Learning Image Restoration without Clean Data" [1].

There are several things different from the original paper
(but not a fatal problem to see how the noise2noise training framework works):
- Training dataset (orignal: ImageNet, this repository: [2])
- Model (original: RED30 [3], this repository: SRResNet [4] or UNet [5])

## Developers
- 이윤호
- 이재우
- 전종혁
- 채현욱

## Dependencies
- Keras >= 2.1.2, TensorFlow, NumPy, OpenCV

## Train Noise2Noise
Any dataset can be used in training and validation instead of the above dataset.

### Train Model
Please see `python3 train.py -h` for optional arguments.


#### Train with Gaussian noise
```bash
# train model using (noise, noise) pairs (noise2noise)
python3 train.py --image_dir dataset/291 --test_dir dataset/Set14 --image_size 128 --batch_size 8 --lr 0.001 --output_path gaussian

# train model using (noise, clean) paris (standard training)
python3 train.py --image_dir dataset/291 --test_dir dataset/Set14 --image_size 128 --batch_size 8 --lr 0.001 --target_noise_model clean --output_path clean
```


#### Train with text insertion

```bash
# train model using (noise, noise) pairs (noise2noise)
python3 train.py --image_dir dataset/291 --test_dir dataset/Set14 --image_size 128 --batch_size 8 --lr 0.001 --source_noise_model text,0,50 --target_noise_model text,0,50 --val_noise_model text,25,25 --loss mae --output_path text_noise

# train model using (noise, clean) paris (standard training)
python3 train.py --image_dir dataset/291 --test_dir dataset/Set14 --image_size 128 --batch_size 8 --lr 0.001 --source_noise_model text,0,50 --target_noise_model clean --val_noise_model text,25,25 --loss mae --output_path text_clean
```

#### Train with random-valued impulse noise

```bash
# train model using (noise, noise) pairs (noise2noise)
python3 train.py --image_dir dataset/291 --test_dir dataset/Set14 --image_size 128 --batch_size 8 --lr 0.001 --source_noise_model impulse,0,95 --target_noise_model impulse,0,95 --val_noise_model impulse,70,70 --loss l0 --output_path impulse_noise

# train model using (noise, clean) paris (standard training)
python3 train.py --image_dir dataset/291 --test_dir dataset/Set14 --image_size 128 --batch_size 8 --lr 0.001 --source_noise_model impulse,0,95 --target_noise_model clean --val_noise_model impulse,70,70 --loss l0 --output_path impulse_clean
```

##### Model architectures
With `--model unet`, UNet model can be trained instead of SRResNet.

##### Resume training
With `--weight path/to/weight/file`, training can be resumed with trained weights.


### Noise Models
Using `source_noise_model`, `target_noise_model`, and `val_noise_model` arguments,
arbitrary noise models can be set for source images, target images, and validatoin images respectively.
Default values are taken from the experiment in [1].

- Gaussian noise
  - gaussian,min_stddev,max_stddev (e.g. gaussian,0,50)
- Clean target
  - clean
- Text insertion
  - text,min_occupancy,max_occupancy (e.g. text,0,50)
- Random-valued impulse noise
  - impulse,min_occupancy,max_occupancy (e.g. impulse,0,50)

You can see how these noise models work by:

```bash
python3 noise_model.py --noise_model text,0,95
```

### Results
#### Plot training history

```bash
python3 plot_history.py --input1 gaussian --input2 clean
```
#### Check denoising result

```bash
python3 test_model.py --weight_file [trained_model_path] --image_dir dataset/Set14
```

The detailed options are:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        test image dir (default: None)
  --model MODEL         model architecture ('srresnet' or 'unet') (default:
                        srresnet)
  --weight_file WEIGHT_FILE
                        trained weight file (default: None)
  --test_noise_model TEST_NOISE_MODEL
                        noise model for test images (default: gaussian,25,25)
  --output_dir OUTPUT_DIR
                        if set, save resulting images otherwise show result
                        using imshow (default: None)
```
This script adds noise using `test_noise_model` to each image in `image_dir` and performs denoising.
If you want to perform denoising to already noisy images, use `--test_noise_model clean`.

## References

[1] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, 
T. Aila, "Noise2Noise: Learning Image Restoration without Clean Data," in Proc. of ICML, 2018.

[2] J. Kim, J. K. Lee, and K. M. Lee, "Accurate Image Super-Resolution Using Very Deep Convolutional Networks," in Proc. of CVPR, 2016.

[3] X.-J. Mao, C. Shen, and Y.-B. Yang, "Image
Restoration Using Convolutional Auto-Encoders with
Symmetric Skip Connections," in Proc. of NIPS, 2016.

[4] C. Ledig, et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network," in Proc. of CVPR, 2017.

[5] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in MICCAI, 2015.
