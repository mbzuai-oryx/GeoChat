# Customize Components in GeoChat

This is an initial guide on how to replace the LLMs, visual encoders, etc. with your choice of components.

## LLM

It is quite simple to swap out LLaMA to any other LLMs.  You can refer to our implementation of [`GeoChat_llama.py`](https://github.com/mbzuai-oryx/GeoChat/blob/main/geochat/model/language_model/geochat_llama.py) for an example of how to replace the LLM.

Although it may seem that it still needs ~100 lines of code, most of them are copied from the original `llama.py` from HF.  The only part that is different is to insert some lines for processing the multimodal inputs.

In `forward` function, you can see that we call `self.prepare_inputs_labels_for_multimodal` to process the multimodal inputs.  This function is defined in `GeoChatMetaForCausalLM` and you just need to insert it into the `forward` function of your LLM.

In `prepare_inputs_for_generation` function, you can see that we add `images` to the `model_inputs`.  This is because we need to pass the images to the LLM during generation.

These are basically all the changes you need to make to replace the LLM.

## Visual Encoder

You can check out [`clip_encoder.py`](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_encoder/clip_encoder.py) on how we implement the CLIP visual encoder.

