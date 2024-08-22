# Visual & Text feature
You can DOWNLOAD feature we used from [BaiduYun](https://pan.baidu.com/s/1Vw9IBO-kecMpjBS0Tb-7nQ?pwd=8xkx). 

unzip feature file at 'save' folder or change the path in corresponding list file.


# Install requirements
Run `conda env create -f environment.yml` to install the requirements.

# Run visdom
**!!!VERY IMPORTANT!!!**

Open a separate terminal and run `visdom` after installing the requirements before running the following commands.


# Model weights
Download Model weights from [BaiduYun](https://pan.baidu.com/s/1mi4z6HWVa9v3mPNz5BR0iw?pwd=s6tb).

Put downloaded model weights in 'ckpt' folder.


# Training + Testing
Meanings of the arguments can be seen in `option.py`. 

# Testing only

UCF-Crime
```bash 
python test.py --feat_extractor clip --dataset ucfcrime --feature-group both --fusion concat --aggregate_text --extra_loss --feature-size 768 --batch-size 64 --rgb_list list/ucf-clip-train-large.list --test_rgb_list list/ucf-clip-test-large.list --use_dic_gt --emb_folder full
```

XD
```bash
python test.py --feat_extractor clip --dataset violence --feature-group both --fusion add --aggregate_text --extra_loss --feature-size 768 --batch-size 64 --rgb_list list/violence-clip-large.list --test_rgb_list list/violence-clip-test-large.list 
```


# Acknowledgements
This code is based on [RTFM](https://github.com/tianyu0207/RTFM/) and [TEVAD](https://github.com/coranholmes/TEVAD).
We thank the authors for their great work