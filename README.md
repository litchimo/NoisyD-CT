# NoisyD-CT
Official PyTorch implementation of the paper "Noisy Disentanglement with Tri-stage Training for Noise-Robust Speech Recognition"，submitted to Applied Acoustics.

## Dataset
The entire dataset is divided into two parts: clean speech and noisy speech.

- clean speech: Librispeech dataset
- noisy speech: noise samples from the CHiME-4 dataset and LibriSpeech clean recordings are mixed to generate synthetic noisy speech

The data folder contains three subfolders: train, dev, and test. Each subfolders have several subsets, each subset contains the following files：
* wav.scp: Clean or noisy speech
* text: Ground truth 

If the subfolders is train, the subsets of it include a special file: 
* clean.scp: The clean speech that corresponds one-to-one to the speech in the wav.scp file
