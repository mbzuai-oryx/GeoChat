# GeoChat <img src="images/logo_geochat.png" height="40">: Grounded Large Vision-Language Model for Remote Sensing
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

#### [Kartik Kuckreja](https://www.linkedin.com/in/kartik-kuckreja-930531221/)\*, [Muhammad Sohail Danish](https://www.linkedin.com/in/muhammad-sohail-danish/)\*, [Muzammal Naseer](https://muzammal-naseer.com/), [Abhijit Das](), [Salman Khan](https://salman-h-khan.github.io/) and [Fahad Khan](https://sites.google.com/view/fahadkhans/home)
\* Equally contributing first authors

#### **Mohamed bin Zayed University of AI, Birla Institute of Technology & Science, Australian National University, Linkoping University**

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://mbzuai-oryx.github.io/GeoChat)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2311.15826)
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://youtu.be/KOKtkkKpNDk)
---

## 📢 Latest Updates
- **Nov-28-23**: GeoChat paper is released [arxiv link](https://arxiv.org/abs/2311.15826). 🔥🔥
- 📦 Code, models, and datasets coming soon! 🚀
---

## <img src="images/logo_geochat.png" height="40">Overview

GeoChat is the first grounded Large Vision Language Model, specifically tailored to Remote Sensing(RS) scenarios. Unlike general-domain models, GeoChat excels in handling high-resolution RS imagery, employing region-level reasoning for comprehensive scene interpretation. Leveraging a newly created RS multimodal dataset, GeoChat is fine-tuned using the LLaVA-1.5 architecture. This results in robust zero-shot performance across various RS tasks, including image and region captioning, visual question answering, scene classification, visually grounded conversations, and referring object detection.

---

## 🏆 Contributions

- **RS multimodal instruction following dataset.** We present a novel data generation pipeline, to leverage existing object detection dataset to create short descriptions of the images, followed by using Vicuna-v1.5 to create conversations using the generated text alone. Further, we add visual question-answering and scene classification abilities 
 using their corresponding datasets. This results in a total of 318k instruction pairs for RS domain.
- **GeoChat.** Leveraging our dataset, we finetune LLaVA-1.5 to create the remote sensing-domain vision-language model - GeoChat. Our LoRA fine-tuning is efficient and avoids forgetting the necessary context embedded in fully-tuned LLaVA model, whose MLP projection is trained to align images into the word embedding space of the LLM (Vicuna-v1.5). This allows GeoChat to retain the conversation and instruction following abilities of LLaVA and extend its domain-knowledge to remote sensing tasks.  

- **Evaluation Benchmark.** We also address the lack of evaluation benchmarks to assess the capability of existing VLMs on remote-sensing conversations. To this end, we setup evaluation protocols for conversation grounding in RS, as well as a setup a suite of tasks to allow comparisons with future efforts in this direction. We show various supervised as well as  zero-shot evaluations for different remote sensing tasks, including image captioning, visual question answering and scene classification to demonstrate the generalisability of GeoChat conversational VLM.

---
## 👁️💬 GeoChat : Grounded Large Vision-Language Model for Remote Sensing

GeoChat can accomplish multiple tasks for remote-sensing (RS) image comprehension in a unified framework. Given suitable task tokens and user queries, the model can generate visually grounded responses (text with corresponding object locations - shown on top), visual question answering on images and regions (top left and bottom right, respectively) as well as scene classification (top right) and normal natural language conversations (bottom). This makes it the first RS VLM with grounding capability. 

<p align="center">
  <img src="images/overview2.png" alt="GeoChat Overview">
</p>

---

## 🛰️ GeoChat : Architecture

An overview of GeoChat - the first grounded large vision-language model for remote sensing. Given an image input together with a user query, a visual backbone is first used to encode patch-level tokens at a higher resolution via interpolating positional encodings. A multi-layer perceptron (MLP) is used to adapt vision-tokens to language space suitable for input to a Large Language Model (Vicuna 1.5). Besides visual inputs, region locations can also be input to the model together with task-specific prompts that specify the desired task required by the user. Given this context, the LLM can generate natural language responses interleaved with corresponding object locations. GeoChat can perform multiple tasks as shown on top e.g., scene classification, image/region captioning, VQA and grounded conversations.

<p align="center">
  <img src="images/architecture.png" alt="GeoChat Architectural">
</p>

---

## 🔍 RS Multimodal Instruction Dataset

Types of annotations available in the GeoChat instruction-set. For a given RS image, we obtain object attribute and relationship information, referring expressions and region captions along with their corresponding region annotations (shown over the image). This structured information is used to create the rich instruction-set with a total of 318k image-instruction pairs.

<p align="center">
  <img src="images/dataset.png" alt="Dataset Annotation Pipeline">
</p>



## 🤖 Qualitative results of GeoChat

Qualitative results of GeoChat. (<em>left-right</em>) Results are shown on grounding, referring object detection, and disaster/damage detection. The user can provide task-specific tokens (e.g., <strong>[grounding]</strong>) to shape model responses according to the desired behavior. The model can generate textual responses (<em>right</em>), only visual grounding (<em>center</em>) and both text and object groundings interleaved together (<em>left</em>). The model can also specify object types, object counts, object attributes and object relationships.
<p align="center">
  <img src="images/examples.png" alt="Results_GCG">
</p>

---

## 🤖 Visual Question Answering
Qualitative examples for Visual Question Answering tasks. GeoChat is able to hold multi-turn conversations, based on various types of questions, including presence, count, complex comparisons and so on. It is able to detect objects and hold conversations against low resolution images as well.
<p align="center">
  <img src="images/vqa.jpg" alt="Visual Question Answering">
</p>

---

## 🤖 Scene Classification
Qualitative examples for scene classification. We give the model all the classes from the dataset and ask to choose only one.
<p align="center">
  <img src="images/scene.jpg" alt="Visual Question Answering">
</p>

---

## 🤖 Grounded Description
When asked to describe the image with the special token '[grounding]', GeoChat outputs both the description of the image as well as the bounding boxes for all the objects detected.
<p align="center">
  <img src="images/grounded.jpg" alt="Grounded Description">
</p>

---

## 🤖 Referring Expression
When asked about an object as a referred expression, GeoChat is able to locate it and draw rotated bounding boxes around it correspondingly.
<p align="center">
  <img src="images/ref1.jpg" alt="Referring Expression">
</p>
<p align="center">
  <img src="images/ref_2.jpg" alt="Referring Expression">
</p>

---

## 🤖 Region Caption
Qualitative examples for region-based captioning. Given a bounding box, GeoChat is able to provide brief descriptions about the area or the object covered by the bounding box.
<p align="center">
  <img src="images/iden.jpg" alt="Region Caption">
</p>

---

## 📜 Citation
```bibtex
@misc{kuckreja2023geochat,
      title={GeoChat: Grounded Large Vision-Language Model for Remote Sensing}, 
      author={Kartik Kuckreja and Muhammad Sohail Danish and Muzammal Naseer and Abhijit Das and Salman Khan and Fahad Shahbaz Khan},
      year={2023},
      eprint={2311.15826},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}  
```
## 🙏 Acknowledgement
We are thankful to LLaVA and Vicuna for releasing their models and code as open-source contributions.

---
[<img src="images/IVAL_logo.png" width="200" height="100">](https://www.ival-mbzuai.com)
[<img src="images/Oryx_logo.png" width="100" height="100">](https://github.com/mbzuai-oryx)
[<img src="images/MBZUAI_logo.png" width="360" height="85">](https://mbzuai.ac.ae)
