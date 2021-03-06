# Mechromancy

This project accompanies the Mechromancy music summoning process. Two separate algorithms are used - a Wave-U-Net implementation is used to separate vocals from a mixed track and then a SampleRNN implementation is used to generate music derivative of a desired vocal or music style.

Samples of what this generated audio sounds like can be found in the samples folder.

# Wave-U-Net

## What is the Wave-U-Net?
The Wave-U-Net is a convolutional neural network applicable to audio source separation tasks, which works directly on the raw audio waveform, presented in [this paper](https://arxiv.org/abs/1806.03185).

The Wave-U-Net is an adaptation of the U-Net architecture to the one-dimensional time domain to perform end-to-end audio source separation. Through a series of downsampling and upsampling blocks, which involve 1D convolutions combined with a down-/upsampling process, features are computed on multiple scales/levels of abstraction and time resolution, and combined to make a prediction.

See the diagram below for a summary of the network architecture.

<img src="./waveunet.png" width="500">

## Downloading the pretrained models

Download the pretrained models [here](https://mega.nz/#!sEw2xCqI!FPpSJROvXHi39qq1l4gzoeaPGO4lALAvVuFmnmFLrVw).
Unzip the archive into the ``Wave-U-Net/checkpoints`` subfolder in this repository, so that you have one subfolder for each model (e.g. ``REPO/Wave-U-Net/checkpoints/baseline_stereo``)

## Run pretrained models

For a quick demo on an example song with the pre-trained best vocal separation model (M5-HighSR), one can simply execute

`` python Predict.py with cfg.full_44KHz ``

to separate the song "Mallory" included in this repository's ``Wave-U-Net/audio_examples`` subfolder into vocals and accompaniment. The output will be saved next to the input file.

To apply the pretrained model to any song, simply point to its audio file path using the ``input_path`` parameter:

`` python Predict.py with cfg.full_44KHz input_path="/Users/marco/mechromancy/Wave-U-Net/audio_examples/Jimmy Carter Says YES - Gene Marshall.mp3"``

If you want to save the predictions to a custom folder instead of where the input song is, just add the ``output_path`` parameter:

`` python Predict.py with cfg.full_44KHz input_path="/Users/marco/mechromancy/Wave-U-Net/audio_examples/Jimmy Carter Says YES - Gene Marshall.mp3" output_path="/home/splits" ``

If you want to use other pre-trained models provided (such as the multi-instrument separator), point to the location of the Tensorflow checkpoint file using the ``model_path`` parameter, making sure that the model configuration (here: ``full_multi_instrument``) matches with the model saved in the checkpoint. As an example for the pre-packaged multi-instrument model:

`` python Predict.py with cfg.full_multi_instrument model_path="checkpoints/full_multi_instrument/full_multi_instrument-134067" input_path="/Users/marco/mechromancy/Wave-U-Net/audio_examples/Jimmy Carter Says YES - Gene Marshall.mp3" output_path="/home/splits" ``

# Dadabots SampleRNN 
## Generating Metal, Rock, Punk, Beatbox

Much of this code accompanies the NIPS 2017 paper [Generating Black Metal and Math Rock: Beyond
Bach, Beethoven, and Beatles](http://dadabots.com/nips2017/generating-black-metal-and-math-rock.pdf) and MUME 2018 paper [Generating Albums with SampleRNN to Imitate Metal, Rock, and Punk Bands](http://musicalmetacreation.org/buddydrive/file/carr/)

This early example of neural synthesis is a proof-of-concept for how machine learning can drive new types of music software. Creating music can be as simple as specifying a set of music influences on which a model trains. We demonstrate a method for generating albums that imitate bands in experimental music genres previously unrealized by traditional synthesis techniques
(e.g. additive, subtractive, FM, granular, concatenative). Unlike MIDI and symbolic models, SampleRNN generates raw audio in the time domain. This requirement becomes increasingly important in modern music styles where timbre and space are used compositionally. Long developmental compositions with rapid transitions between sections are possible by increasing the depth of the network beyond the number used for speech datasets. We are delighted by the unique characteristic artifacts of neural synthesis.

# SampleRNN (Dadabots fork)

Original SampleRNN paper [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://openreview.net/forum?id=SkxKPDv5xl). 

## Features
- Load a dataset of audio
- Train a model on that audio to predict "given what just happened, what comes next?"
- Generate new audio by iteratively choosing "what next comes" indefinitely 

## Modifications from original code:
- Auto-preprocessing (audio conversion, concatenation, chunking, and saving .npy files). We find splitting an album into 32 00 overlapping chunks of 8 seconds to give us good results. 
- New scripts for generating 100s of audio examples in parallel from a trained net.
- New scripts for different sample rates are available (16k, 32k). 32k audio sounds better, but the nets take longer to train, and they don't learn structure as well as 16k.
- Any processed datasets can be loaded into the two-tier network via arguments. This significantly speeds up the workflow without having to change code. 
- Sampling is picked from distribution (not argmax). This makes better sense because certain sounds (noise, texture, the "s" sound in speech) are inherently stochastic. Also this is significant for avoiding traps (the generated audio gets stuck in a loop). 
- Wny number of RNN layers is now possible (until you run out of memory). This was significant to getting good results. The original limit was insufficient for music, we get good results with 5 layers. 
- Local conditioning. Although we haven't fully researched the possibilities of local conditioning, we coded it in. 
- Fix bad amplitude normalization causing DC offsets (see [issue](https://github.com/soroushmehr/sampleRNN_ICLR2017/issues/24)) 

## Dependencies

The original code lists:
- cuDNN 5105
- Python 2.7.12
- Numpy 1.11.1
- Theano 0.8.2 
- Lasagne 0.2.dev1
- ffmpeg (libav-tools)

But we get much faster code using the next generation of GPU architecture with:
- CUDA 9.2
- cuDNN 8.0
- Theano 1.0
- NVIDIA V100 GPU

## Setup

A detailed description of how we setup this code on Ubuntu 16.04 with NVIDIA 100 GPU can be found here. 

[DETAILED SETUP INSTRUCTIONS](https://github.com/Cortexelus/dadabots_sampleRNN/wiki/Installing-Dadabots-SampleRNN-on-Ubuntu)



## Datasets
To create a new dataset, place your audio here:
```
datasets/music/downloads/
```
then run the new experiment python script located in the datasets/music directory:

16k sample rate: 
```
cd datasets/music/
sudo python new_experiment16k.py jeromes downloads/
```

32k sample rate: 
```
cd datasets/music/
sudo python new_experiment32k.py jeromes downloads/
```

## Training
To train a model on an existing dataset with accelerated GPU processing, you need to run following lines from the root of `dadabots_sampleRNN` folder which corresponds to the best found set of hyper-paramters.

Mission control center:
```
$ pwd
/home/ubuntu/https://github.com/landmarco/mechromancy
```

### Training SampleRNN (2-tier)
```
$ python models/two_tier/two_tier32k.py -h
usage: two_tier.py [-h] [--exp EXP] --n_frames N_FRAMES --frame_size
                   FRAME_SIZE --weight_norm WEIGHT_NORM --emb_size EMB_SIZE
                   --skip_conn SKIP_CONN --dim DIM --n_rnn {1,2,3,4,5}
                   --rnn_type {LSTM,GRU} --learn_h0 LEARN_H0 --q_levels
                   Q_LEVELS --q_type {linear,a-law,mu-law} --which_set
                   {...} --batch_size {64,128,256} [--debug]
                   [--resume]

two_tier.py No default value! Indicate every argument.

optional arguments:
  -h, --help            show this help message and exit
  --exp EXP             Experiment name (name it anything you want)
  --n_frames N_FRAMES   How many "frames" to include in each Truncated BPTT
                        pass
  --frame_size FRAME_SIZE
                        How many samples per frame
  --weight_norm WEIGHT_NORM
                        Adding learnable weight normalization to all the
                        linear layers (except for the embedding layer)
  --emb_size EMB_SIZE   Size of embedding layer (0 to disable)
  --skip_conn SKIP_CONN
                        Add skip connections to RNN
  --dim DIM             Dimension of RNN and MLPs
  --n_rnn {1,2,3,4,5,6,7,8,9,10,11,12,n,...}
					 	Number of layers in the stacked RNN
  --rnn_type {LSTM,GRU}
                        GRU or LSTM
  --learn_h0 LEARN_H0   Whether to learn the initial state of RNN
  --q_levels Q_LEVELS   Number of bins for quantization of audio samples.
                        Should be 256 for mu-law.
  --q_type {linear,a-law,mu-law}
                        Quantization in linear-scale, a-law-companding, or mu-
                        law compandig. With mu-/a-law quantization level shoud
                        be set as 256
  --which_set {...}
                        The name of the dataset you created. In the above example "krallice"
  --batch_size {64,128,256}
                        size of mini-batch
  --debug               Debug mode
  --resume              Resume the same model from the last checkpoint. Order
                        of params are important. [for now]
```


If you're using cuda9 with v100 gpus, you need "device=cuda0" 

If you're using cuda8 with K80 gpus or earlier, you may need "device=gpu0" instead

If you have 8 GPUs, you can run up to 8 experiments in parallel, by setting device to cuda0, cuda1, cuda2, cuda3... cuda7


#### Best hyperparameters

Here are the hyperparameters I used for training on Jeromes Dream, and would applicable for most forms of extreme music.

```
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32 python -u models/two_tier/two_tier16k.py --exp i_dream_of_jeromes_dream --n_frames 64 --frame_size 16 --emb_size 256 --skip_conn True --dim 1024 --n_rnn 5 --rnn_type LSTM --q_levels 256 --q_type mu-law --batch_size 128 --weight_norm True --learn_h0 False --which_set i_dream_of_jeromes_dream
```

## Generating

Generate 100 songs (4 minutes each) from a trained 32k model:
```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python -u models/two_tier/two_tier_generate32k.py --exp i_dream_of_jeromes_dream_experiment --n_frames 64 --frame_size 16 --emb_size 256 --skip_conn True --dim 1024 --n_rnn 5 --rnn_type LSTM --q_levels 256 --q_type mu-law --batch_size 128 --weight_norm True --learn_h0 False --which_set i_dream_of_jeromes_dream --n_secs 240 --n_seqs 100
```

All the parameters have to be the same as when you trained it. Notice we're calling `two_tier_generate32k.py` with two new flags `--n_secs` and `--n_seqs` 

It will take just as much time to generate 100 songs as 5, because they are created in parallel (up to a hardware memory limit). 

This will generate from the latest checkpoint. However, often the latest checkpoint does not always create the best music. Instead we listen to the test audio generated at each checkpoint, choose our favorite checkpoint, and delete the newer checkpoints, before generating a huge batch with this script. 


## The Summoning

The suggested workflow is to for human curation, sorting through the generated audio and selecting striking parts and assembling them together, like organizing a playlist.

## Reference
Many thanks to CJ Carr [[github]](https://github.com/Cortexelus) [[website]](http://cortexel.us) and Zack Zukowski [[github]](https://github.com/ZVK) [[website]](http://zackzukowski.com/) for their work on Dadabots which inspired this project. Thanks to Soroush Mehri, Kundan Kumar, Ishaan Gulrajani, Rithesh Kumar, Shubham Jain, Jose Sotelo, Aaron Courville, and Yoshua Bengio for their work on the SampleRNN implementation, and to Daniel Stoller, Sebastian Ewert, and Simon Dixon for their work on Wave-U-Net. Their papers are cited below:  

Generating Albums with SampleRNN to Imitate Metal, Rock, and Punk Bands. CJ Carr, Zack Zukowski (MUME 2018).

SampleRNN: An Unconditional End-to-End Neural Audio Generation Model. Soroush Mehri, Kundan Kumar, Ishaan Gulrajani, Rithesh Kumar, Shubham Jain, Jose Sotelo, Aaron Courville, Yoshua Bengio, 5th International Conference on Learning Representations (ICLR 2017).

Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation. Daniel Stoller, Sebastian Ewert, Simon Dixon (SiSec 2018).

## License

This documentation licensed CC-BY 4.0

The source code is licensed Apache 2.0

I also dedicate this project in the spirit of the poet, retold by Bascom Lamar Lunsford and originally written by Henry Wadsworth Longfellow:  

The Arrow and the Song

I shot an arrow into the air,  
It fell to earth, I know not where;  
For long years after, still unbroke,  
I found that arrow in the heart of an oak.

I breathe the song into the air,  
It came to earth, I know not where;  
But for long years after, from beginning to end,  
I found that song in the heart of a friend.
