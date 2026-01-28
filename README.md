# The hyena-dna model trainable on Intel Arc GPU
Modified hyena-dna, compatible with training on Intel XPU.
## Environment
### Hardware
CPU: Intel Core i9 10940x
DDR4:128GB
GPU: Intel Arc A770 16GB


### Software
Operating System: Fedora Workstation 43 x86_64
Intel oneAPI = 2025.3
Python = 3.11
Pytorch = 2.8.10 + xpu

# Dependencies
## Install Intel oneAPI XPU support
Download and install Intel® Deep Learning Essentials
```
https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=dl-essentials&dl-essentials-os=linux&dl-lin=offline
```

## clone repo, cd into it
```bash
git clone --recurse-submodules https://github.com/N-A-U/hyena-dna-xpu.git && cd hyena-dna-xpu
```
## create a conda environment, with Python 3.11
```bash
conda create -n hyena-dna-idp intelpython3_full python=3.11 -c https://software.repos.intel.com/python/conda -c conda-forge --override-channels
```
## The repo is developed with Pytorch 2.8, using Intel xpu
```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
python -m pip install intel-extension-for-pytorch==2.8.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
python -m pip install oneccl_bind_pt==2.8.0+xpu --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```
## install requirements:
```bash
pip install -r requirements.txt
```



## Experiments

We share our training and dataloading code for pretraining on the human reference genome (HG38), fine-tuning on a number of downstreams, and examples of our in-context learning variants using soft prompt tokens and instruction fine-tuning. You'll need to download and preprocess on your own for now, we'll share our steps for those later.

In general, get comfortable with the configs in `configs/experiments/hg38`, all our (sample) experiment settings are there.

### Pretraining on Human Reference Genome
<a name="pretraining"></a>

First step is download the Human Reference Genome data. It's comprised of 2 files, 1 with all the sequences (the .fasta file), and with the intervals we use (.bed file).

The file structure should look like
```
data
|-- hg38/
    |-- hg38.ml.fa
    |-- human-sequences.bed

```

- Download fasta (.fa format) file (of the entire human genome) into hyena-dna/data/hg38.  ~24 chromosomes in the whole genome (merged into 1 file), each chromosome is a continuous sequence, basically. Then download the .bed file with sequence intervals (contains chromosome name, start, end, split, which then allow you to retrieve from the fasta file)  

```bash
mkdir -p data/hg38/
curl https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz > data/hg38/hg38.ml.fa.gz
curl https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed > data/hg38/human-sequences.bed
```

launch pretraining run  

```bash
python train_xpu.py \
  wandb=null \
  experiment=hg38/hg38_hyena \
  model.d_model=128 \
  model.n_layer=2 \
  dataset.batch_size=512 \
  train.global_batch_size=512 \
  dataset.max_length=1024 \
  optimizer.lr=6e-4 \
  trainer.devices=1 \
  dataset.num_workers=2 \
  train.validate_at_start=false
```

