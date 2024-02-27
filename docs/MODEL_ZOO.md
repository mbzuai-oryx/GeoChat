# Model Zoo

| Base LLM | Vision Encoder | Pretrain Data | Pretraining schedule | Finetuning Data | Finetuning schedule | Download |
|----------|----------------|---------------|----------------------|-----------------|--------------------|------------------
| Vicuna-13B-v1.3 | CLIP-L-336px(extended to 504) | LCS-558K | 1e | Geochat_Instruct | proj-1e, lora-1e | [LoRA-Merged](https://huggingface.co/MBZUAI/geochat-7B) |

## Projector weights
We use the projector from LlaVA-1.5 for initialization. [Link](https://huggingface.co/liuhaotian/llava-v1.5-7b-lora)

**NOTE**: When you use our pretrained projector for visual instruction tuning, it is very important to **use the same base LLM and vision encoder** as the one we used for pretraining the projector. Otherwise, the performance will be very bad.

When using these projector weights to instruction tune your LMM, please make sure that these options are correctly set as follows,

```Shell
--mm_use_im_start_end False
--mm_use_im_patch_token False
```

