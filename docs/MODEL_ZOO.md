# Model Zoo

| Version | Size | Schedule | Checkpoint | VQAv2 | GQA | VizWiz | SQA | T-VQA | POPE | MME | MM-Bench | MM-Bench-CN | SEED | LLaVA-Bench-Wild | MM-Vet |
|----------|----------|-----------|-----------|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5 | 7B | full_ft-1e | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b), [logs](https://api.wandb.ai/links/lht/6orh56wc) | 78.5 | 62.0 | 50.0 | 66.8 | 58.2 | 85.9 | 1510.7 | 64.3 | 58.3 | 58.6 | 65.4 | 31.1 |
| LLaVA-1.5 | 13B | full_ft-1e | [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b), [logs](https://api.wandb.ai/links/lht/6orh56wc) | 80.0 | 63.3 | 53.6 | 71.6 | 61.3 | 85.9 | 1531.3 | 67.7 | 63.6 | 61.6 | 72.5 | 36.1 |
| LLaVA-1.5 | 7B | lora-1e | coming soon |
| LLaVA-1.5 | 13B | lora-1e | coming soon |


| Base LLM | Vision Encoder | Pretrain Data | Pretraining schedule | Finetuning Data | Finetuning schedule | Download |
|----------|----------------|---------------|----------------------|-----------------|--------------------|------------------|--------------------|---------------------|---------------------|---------------------|
| Vicuna-13B-v1.3 | CLIP-L-336px(extended to 504) | LCS-558K | 1e | Geochat_Instruct | proj-1e, lora-1e | [LoRA-Merged](https://huggingface.co/MBZUAI/geochat-7B) |

## Projector weights
We use the projector from LlaVA-1.5 for initialization. [Link](https://huggingface.co/liuhaotian/llava-v1.5-7b-lora)

**NOTE**: When you use our pretrained projector for visual instruction tuning, it is very important to **use the same base LLM and vision encoder** as the one we used for pretraining the projector. Otherwise, the performance will be very bad.

When using these projector weights to instruction tune your LMM, please make sure that these options are correctly set as follows,

```Shell
--mm_use_im_start_end False
--mm_use_im_patch_token False
```

