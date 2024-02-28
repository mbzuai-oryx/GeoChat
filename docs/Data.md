##  Finetuning Data
We use GeoChat-Instruct to finetune our model. The instruction following dataset is present in GeoChat_Instruct.json and the images are present in the [huggingface repo](https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct). The images are split into multiple files. Download the separate files in the same folder and run the following script to merge them. 

```Shell
cat images_parta* > images.zip
```

Unzip the images in a folder and provide the folder path in training and evaluation scripts.

| Data file name | Size |
| --- | ---: |
| [GeoChat_Instruct](https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct/blob/main/GeoChat_Instruct.json) | 263 MB |

## Pretraining Dataset
We use the same pretraining dataset as of LlaVA-v1.5.
The pretraining dataset used in this release is a subset of CC-3M dataset, filtered with a more balanced concept coverage distribution.  Please see [here](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) for a detailed description of the dataset structure and how to download the images.

If you already have CC-3M dataset on your disk, the image names follow this format: `GCC_train_000000000.jpg`.  You may edit the `image` field correspondingly if necessary.

| Data | Chat File | Meta Data | Size |
| --- |  --- |  --- | ---: |
| CC-3M Concept-balanced 595K | [chat.json](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/chat.json) | [metadata.json](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/metadata.json) | 211 MB
| LAION/CC/SBU BLIP-Caption Concept-balanced 558K | [blip_laion_cc_sbu_558k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/blip_laion_cc_sbu_558k.json) | [metadata.json](#) | 181 MB

