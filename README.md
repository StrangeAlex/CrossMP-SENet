# CrossMP-SENet - Usage Guide

## 1. Setup

For inference part only, install best trained model via [this link](https://drive.google.com/drive/folders/1Svew4M6rAXEmxzAUgsQcXQ1BvjSs1ci5?usp=sharing).

Install the dependencies with:

```bash
pip install -r requirements.txt
```

---

## 2. Data Preparation

Organize your data under a root directory (e.g. `VoiceBank+DEMAND`) with the following structure:

```
VoiceBank+DEMAND/
├── wav_clean/               # clean .wav files
├── wav_noisy/               # corresponding noisy .wav files
├── testset_noisy/           # noisy test samples (for inference)
├── training.txt             # training file list, format: filename|*
└── test.txt                 # validation file list, format: filename|*
```

Each line in `training.txt` or `test.txt` should list a filename (without extension) and a dummy label (e.g. `p257_001|0`).

---

## 3. Training

Start training using:

```bash
python train.py --config config.json --checkpoint_path checkpoints
```

You can also customize arguments like:

* `--training_epochs`: number of epochs (default: 500)
* `--stdout_interval`: steps between printed logs (default: 2500)
* `--validation_interval`: validation frequency (default: 1500)

The training loop periodically saves generator (`g_*.pt`) and discriminator (`do_*.pt`) checkpoints in the subfolder of the checkpoint path.

---

## 4. Inference

After training, run inference using:

```bash
python inference.py --checkpoint_file checkpoints/g_XXXXX
```
