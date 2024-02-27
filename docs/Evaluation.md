# Evaluation

We evaluate GeoChat on a variety of tasks, including scene classification, region captioning, visual grounding, grounding description and VQA.
Converted files in the input format for GeoChat are available at [GeoChat-Bench](https://huggingface.co/datasets/MBZUAI/GeoChat-Bench/tree/main)


Below we provide a general guideline for evaluating datasets.

1. LRBEN/HRBEN.
Images and ground truth for evaluation need to be downloaded from the following sources: [LRBEN](https://zenodo.org/records/6344334), [HRBEN](https://zenodo.org/records/6344367)
Give the path to the extracted image folder in the evaluation script. We add the following text after each question during our evaluation.
```
<question>
Answer the question using a single word or phrase.
```
```Shell
python geochat/eval/batch_geochat_vqa.py \
    --model-path /path/to/model \
    --question-file path/to/jsonl/file \
    --answer-file path/to/output/jsonl/file \
    --image_folder path/to/image/folder/
```
2. Scene Classification.
Download the images from the following sources, [UCmerced](http://weegee.vision.ucmerced.edu/datasets/landuse.html), [AID](https://drive.google.com/drive/folders/1-1D9DrYYWMGuuxx-qcvIIOV1oUkAVf-M). We add the following text after each question during our evaluation.
```
<question>
Classify the image from the following classes. Answer in one word or a short phrase.
```
```Shell
python geochat/eval/batch_geochat_scene.py \
    --model-path /path/to/model \
    --question-file path/to/jsonl/file \
    --answer-file path/to/output/jsonl/file \
    --image_folder path/to/image/folder/
```

3. Region-Captioning/Visual grounding.

The evaluation images are present in the image.zip folder in [GeoChat_Instruct](https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct/blob/main/images.zip). 
```Shell
python geochat/eval/batch_geochat_grounding.py \
    --model-path /path/to/model \
    --question-file path/to/jsonl/file \
    --answer-file path/to/output/jsonl/file \
    --image_folder path/to/image/folder/
```

```Shell
python geochat/eval/batch_geochat_referring.py \
    --model-path /path/to/model \
    --question-file path/to/jsonl/file \
    --answer-file path/to/output/jsonl/file \
    --image_folder path/to/image/folder/
```