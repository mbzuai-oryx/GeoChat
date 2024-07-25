import argparse
from functools import partial
import os
import random
from collections import defaultdict

import cv2
import re
import math
import numpy as np
from PIL import Image
import torch
import html
import gradio as gr

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from geochat.conversation import conv_templates, Chat
from geochat.model.builder import load_pretrained_model
from geochat.mm_utils import  get_model_name_from_path


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    # parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu-id", type=str,default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    # args = parser.parse_args()
    args = parser.parse_args()
    return args


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cudnn.benchmark = False
cudnn.deterministic = True

print('Initializing Chat')
args = parse_args()
# cfg = Config(args)

model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
assert tokenizer is not None, f'Model probably did not load: {model_name}'
assert model is not None, f'Model probably did not load: {model_name}'
assert image_processor is not None, f'Model probably did not load: {model_name}'
assert context_len is not None, f'Model probably did not load: {model_name}'

device = 'cuda:{}'.format(args.gpu_id)

# model_config = cfg.model_cfg
# model_config.device_8bit = args.gpu_id
# model_cls = registry.get_model_class(model_config.arch)
# model = model_cls.from_config(model_config).to(device)
bounding_box_size = 100

# vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
# vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

model = model.eval()

CONV_VISION = conv_templates['llava_v1'].copy()

def bbox_and_angle_to_polygon(x1, y1, x2, y2, a):
    # Calculate center coordinates
    x_ctr = (x1 + x2) / 2
    y_ctr = (y1 + y2) / 2
    
    # Calculate width and height
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    
    # Calculate the angle in radians
    angle_rad = math.radians(a)
    
    # Calculate coordinates of the four corners of the rotated bounding box
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    x1_rot = cos_a * (-w / 2) - sin_a * (-h / 2) + x_ctr
    y1_rot = sin_a * (-w / 2) + cos_a * (-h / 2) + y_ctr
    
    x2_rot = cos_a * (w / 2) - sin_a * (-h / 2) + x_ctr
    y2_rot = sin_a * (w / 2) + cos_a * (-h / 2) + y_ctr
    
    x3_rot = cos_a * (w / 2) - sin_a * (h / 2) + x_ctr
    y3_rot = sin_a * (w / 2) + cos_a * (h / 2) + y_ctr
    
    x4_rot = cos_a * (-w / 2) - sin_a * (h / 2) + x_ctr
    y4_rot = sin_a * (-w / 2) + cos_a * (h / 2) + y_ctr
    
    # Return the polygon coordinates
    polygon_coords = np.array((x1_rot, y1_rot, x2_rot, y2_rot, x3_rot, y3_rot, x4_rot, y4_rot))
    
    return polygon_coords

def rotate_bbox(top_right, bottom_left, angle_degrees):
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Calculate the center of the rectangle
    center = ((top_right[0] + bottom_left[0]) / 2, (top_right[1] + bottom_left[1]) / 2)

    # Calculate the width and height of the rectangle
    width = top_right[0] - bottom_left[0]
    height = top_right[1] - bottom_left[1]

    # Create a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1)

    # Create an array of the rectangle corners
    rectangle_points = np.array([[bottom_left[0], bottom_left[1]],
                                 [top_right[0], bottom_left[1]],
                                 [top_right[0], top_right[1]],
                                 [bottom_left[0], top_right[1]]], dtype=np.float32)

    # Rotate the rectangle points
    rotated_rectangle = cv2.transform(np.array([rectangle_points]), rotation_matrix)[0]

    return rotated_rectangle
def extract_substrings(string):
    # first check if there is no-finished bracket
    index = string.rfind('}')
    if index != -1:
        string = string[:index + 1]

    pattern = r'<p>(.*?)\}(?!<)'
    matches = re.findall(pattern, string)
    substrings = [match for match in matches]

    return substrings


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou


def save_tmp_img(visual_img):
    file_name = "".join([str(random.randint(0, 9)) for _ in range(5)]) + ".jpg"
    file_path = "/tmp/gradio" + file_name
    visual_img.save(file_path)
    return file_path


def mask2bbox(mask):
    if mask is None:
        return ''
    mask = mask.resize([100, 100], resample=Image.NEAREST)
    mask = np.array(mask)[:, :, 0]

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.sum():
        # Get the top, bottom, left, and right boundaries
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = '{{<{}><{}><{}><{}>}}'.format(cmin, rmin, cmax, rmax)
    else:
        bbox = ''

    return bbox


def escape_markdown(text):
    # List of Markdown special characters that need to be escaped
    md_chars = ['<', '>']

    # Escape each special character
    for char in md_chars:
        text = text.replace(char, '\\' + char)

    return text


def reverse_escape(text):
    md_chars = ['\\<', '\\>']

    for char in md_chars:
        text = text.replace(char, char[1:])

    return text


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (210, 210, 0),
    (255, 0, 255),
    (0, 255, 255),
    (114, 128, 250),
    (0, 165, 255),
    (0, 128, 0),
    (144, 238, 144),
    (238, 238, 175),
    (255, 191, 0),
    (0, 128, 0),
    (226, 43, 138),
    (255, 0, 255),
    (0, 215, 255),
]

color_map = {
    f"{color_id}": f"#{hex(color[2])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[0])[2:].zfill(2)}" for
    color_id, color in enumerate(colors)
}

used_colors = colors


def visualize_all_bbox_together(image, generation):
    if image is None:
        return None, ''

    generation = html.unescape(generation)

    image_width, image_height = image.size
    image = image.resize([500, int(500 / image_width * image_height)])
    image_width, image_height = image.size

    string_list = extract_substrings(generation)
    if string_list:  # it is grounding or detection
        mode = 'all'
        entities = defaultdict(list)
        i = 0
        j = 0
        for string in string_list:
            try:
                obj, string = string.split('</p>')
            except ValueError:
                print('wrong string: ', string)
                continue
            if "}{" in string:
                string=string.replace("}{","}<delim>{")
            bbox_list = string.split('<delim>')
            flag = False
            for bbox_string in bbox_list:
                integers = re.findall(r'-?\d+', bbox_string)
                if len(integers)==4:
                    angle=0
                else:
                    angle=integers[4]
                integers=integers[:-1]
                
                if len(integers) == 4:
                    x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
                    left = x0 / bounding_box_size * image_width
                    bottom = y0 / bounding_box_size * image_height
                    right = x1 / bounding_box_size * image_width
                    top = y1 / bounding_box_size * image_height

                    entities[obj].append([left, bottom, right, top,angle])

                    j += 1
                    flag = True
            if flag:
                i += 1
    else:
        integers = re.findall(r'-?\d+', generation)
        # if len(integers)==4:
        angle=0
        # else:
            # angle=integers[4]
        integers=integers[:-1]
        if len(integers) == 4:  # it is refer
            mode = 'single'

            entities = list()
            x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
            left = x0 / bounding_box_size * image_width
            bottom = y0 / bounding_box_size * image_height
            right = x1 / bounding_box_size * image_width
            top = y1 / bounding_box_size * image_height
            entities.append([left, bottom, right, top,angle])
        else:
            # don't detect any valid bbox to visualize
            return None, ''

    if len(entities) == 0:
        return None, ''

    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)

    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):

        image_tensor = image.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = T.ToPILImage()(image_tensor)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invalid image format, {type(image)} for {image}")

    indices = list(range(len(entities)))

    new_image = image.copy()

    previous_bboxes = []
    # size of text
    text_size = 0.4
    # thickness of text
    text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = 2
    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 2

    # num_bboxes = sum(len(x[-1]) for x in entities)
    used_colors = colors  # random.sample(colors, k=num_bboxes)

    color_id = -1
    for entity_idx, entity_name in enumerate(entities):
        if mode == 'single' or mode == 'identify':
            bboxes = entity_name
            bboxes = [bboxes]
        else:
            bboxes = entities[entity_name]
        color_id += 1
        for bbox_id, (x1_norm, y1_norm, x2_norm, y2_norm,angle) in enumerate(bboxes):
            skip_flag = False
            orig_x1, orig_y1, orig_x2, orig_y2,angle = int(x1_norm), int(y1_norm), int(x2_norm), int(y2_norm), int(angle)

            color = used_colors[entity_idx % len(used_colors)] # tuple(np.random.randint(0, 255, size=3).tolist())
            top_right=(orig_x1,orig_y1)
            bottom_left=(orig_x2,orig_y2)
            angle=angle
            rotated_bbox = rotate_bbox(top_right, bottom_left, angle)
            new_image=cv2.polylines(new_image, [rotated_bbox.astype(np.int32)], isClosed=True,thickness=2, color=color)

            # new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

            if mode == 'all':
                l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

                x1 = orig_x1 - l_o
                y1 = orig_y1 - l_o

                if y1 < text_height + text_offset_original + 2 * text_spaces:
                    y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                    x1 = orig_x1 + r_o

                # add text background
                (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size,
                                                               text_line)
                text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (
                            text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

                for prev_bbox in previous_bboxes:
                    if computeIoU((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']) > 0.95 and \
                            prev_bbox['phrase'] == entity_name:
                        skip_flag = True
                        break
                    while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']):
                        text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
                        text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
                        y1 += (text_height + text_offset_original + 2 * text_spaces)

                        if text_bg_y2 >= image_h:
                            text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
                            text_bg_y2 = image_h
                            y1 = image_h
                            break
                if not skip_flag:
                    alpha = 0.5
                    for i in range(text_bg_y1, text_bg_y2):
                        for j in range(text_bg_x1, text_bg_x2):
                            if i < image_h and j < image_w:
                                if j < text_bg_x1 + 1.35 * c_width:
                                    # original color
                                    bg_color = color
                                else:
                                    # white
                                    bg_color = [255, 255, 255]
                                new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(
                                    np.uint8)

                    cv2.putText(
                        new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces),
                        cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA
                    )

                    previous_bboxes.append(
                        {'bbox': (text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), 'phrase': entity_name})

    if mode == 'all':
        def color_iterator(colors):
            while True:
                for color in colors:
                    yield color

        color_gen = color_iterator(colors)

        # Add colors to phrases and remove <p></p>
        def colored_phrases(match):
            phrase = match.group(1)
            color = next(color_gen)
            return f'<span style="color:rgb{color}">{phrase}</span>'

        generation = re.sub(r'{<\d+><\d+><\d+><\d+>}|<delim>', '', generation)
        generation_colored = re.sub(r'<p>(.*?)</p>', colored_phrases, generation)
    else:
        generation_colored = ''

    pil_image = Image.fromarray(new_image)
    return pil_image, generation_colored


def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return (
        None,
        gr.update(value=None, interactive=True),
        gr.update(placeholder='Upload your image and chat', interactive=True),
        chat_state,
        img_list
    )


def image_upload_trigger(upload_flag, replace_flag, img_list):
    # set the upload flag to true when receive a new image.
    # if there is an old image (and old conversation), set the replace flag to true to reset the conv later.
    upload_flag = 1
    if img_list:
        replace_flag = 1
    return upload_flag, replace_flag


def gradio_ask(user_message, chatbot, chat_state, gr_img, img_list, upload_flag, replace_flag):
    if len(user_message) == 0:
        text_box_show = 'Input should not be empty!'
    else:
        text_box_show = ''

    if isinstance(gr_img, dict):
        gr_img, mask = gr_img['image'], gr_img['mask']
    else:
        mask = None

    if '[identify]' in user_message:
        # check if user provide bbox in the text input
        integers = re.findall(r'-?\d+', user_message)
        if len(integers) != 4:  # no bbox in text
            bbox = mask2bbox(mask)
            user_message = user_message + bbox

    if chat_state is None:
        chat_state = CONV_VISION.copy()

    if upload_flag:
        if replace_flag:
            chat_state = CONV_VISION.copy()  # new image, reset everything
            replace_flag = 0
            chatbot = []
        img_list = []
        llm_message = chat.upload_img(gr_img, chat_state, img_list)
        upload_flag = 0

    chat.ask(user_message, chat_state)

    chatbot = chatbot + [[user_message, None]]

    if '[identify]' in user_message:
        visual_img, _ = visualize_all_bbox_together(gr_img, user_message)
        if visual_img is not None:
            file_path = save_tmp_img(visual_img)
            chatbot = chatbot + [[(file_path,), None]]

    return text_box_show, chatbot, chat_state, img_list, upload_flag, replace_flag


# def gradio_answer(chatbot, chat_state, img_list, temperature):
#     llm_message = chat.answer(conv=chat_state,
#                               img_list=img_list,
#                               temperature=temperature,
#                               max_new_tokens=500,
#                               max_length=2000)[0]
#     chatbot[-1][1] = llm_message
#     return chatbot, chat_state


def gradio_stream_answer(chatbot, chat_state, img_list, temperature):
    if len(img_list) > 0:
        if not isinstance(img_list[0], torch.Tensor):
            chat.encode_img(img_list)
    streamer = chat.stream_answer(conv=chat_state,
                                  img_list=img_list,
                                  temperature=temperature,
                                  max_new_tokens=500,
                                  max_length=2000)
    # chatbot[-1][1] = output
    # chat_state.messages[-1][1] = '</s>'
    
    output = ''
    for new_output in streamer:
        # print(new_output)
        output=output+new_output
    print(output)
    # if "{" in output:
    #     chatbot[-1][1]="Grounding and referring expression is still under work."
    # else:
    output = escape_markdown(output)
        # output += escapped
    chatbot[-1][1] = output
    yield chatbot, chat_state
    chat_state.messages[-1][1] = '</s>'
    return chatbot, chat_state


def gradio_visualize(chatbot, gr_img):
    if isinstance(gr_img, dict):
        gr_img, mask = gr_img['image'], gr_img['mask']

    unescaped = reverse_escape(chatbot[-1][1])
    visual_img, generation_color = visualize_all_bbox_together(gr_img, unescaped)
    if visual_img is not None:
        if len(generation_color):
            chatbot[-1][1] = generation_color
        file_path = save_tmp_img(visual_img)
        chatbot = chatbot + [[None, (file_path,)]]

    return chatbot


def gradio_taskselect(idx):
    prompt_list = [
        '',
        'Classify the image in the following classes: ',
        '[identify] what is this ',
    ]
    instruct_list = [
        '**Hint:** Type in whatever you want',
        '**Hint:** Type in the classes you want the model to classify in',
        '**Hint:** Draw a bounding box on the uploaded image then send the command. Click the "clear" botton on the top right of the image before redraw',
    ]
    return prompt_list[idx], instruct_list[idx]




chat = Chat(model, image_processor, tokenizer, device=device)


title = """<h1 align="center">GeoChat Demo</h1>"""
description = 'Welcome to Our GeoChat Chatbot Demo!'
article = """<div style="display: flex;"><p style="display: inline-block;"><a href='https://mbzuai-oryx.github.io/GeoChat'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p style="display: inline-block;"><a href='https://arxiv.org/abs/2311.15826'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p><p style="display: inline-block;"><a href='https://github.com/mbzuai-oryx/GeoChat/tree/main'><img src='https://img.shields.io/badge/GitHub-Repo-blue'></a></p><p style="display: inline-block;"><a href='https://youtu.be/KOKtkkKpNDk?feature=shared'><img src='https://img.shields.io/badge/YouTube-Video-red'></a></p></div>"""
# article = """<p><a href='https://minigpt-v2.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p>"""

introduction = '''
1. Identify: Draw the bounding box on the uploaded image window and CLICK **Send** to generate the bounding box. (CLICK "clear" button before re-drawing next time).
2. No Tag: Input whatever you want and CLICK **Send** without any tagging

You can also simply chat in free form!
'''


text_input = gr.Textbox(placeholder='Upload your image and chat', interactive=True, show_label=False, container=False,
                        scale=12)
with gr.Blocks() as demo:
    # gr.Markdown(title)
    # gr.Markdown(description)
    # gr.Markdown(article)

    with gr.Row():
        with gr.Column(scale=0.5):
            #image = gr.Image(type="pil", tool='sketch', brush_radius=20)
            image = gr.Image(type="pil")

            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.6,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

            clear = gr.Button("Restart")

            gr.Markdown(introduction)

        with gr.Column():
            chat_state = gr.State(value=None)
            img_list = gr.State(value=[])
            chatbot = gr.Chatbot(label='GeoChat')

            dataset = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                samples=[['No Tag'], ['Scene Classification'],['Identify']],
                type="index",
                label='Task Shortcuts',
            )
            task_inst = gr.Markdown('**Hint:** Upload your image and chat')
            with gr.Row():
                text_input.render()
                send = gr.Button("Send", variant='primary', size='sm', scale=1)

    upload_flag = gr.State(value=0)
    replace_flag = gr.State(value=0)
    image.upload(
        image_upload_trigger,
        [upload_flag, replace_flag, img_list],
        [upload_flag, replace_flag]
    )

    def example_trigger(text_input, image):
        # set the upload flag to true when receive a new image.
        # if there is an old image (and old conversation), set the replace flag to true to reset the conv later.
        upload_flag = 1
        if img_list or replace_flag == 1:
            replace_flag = 1

        return upload_flag, replace_flag

    with gr.Row():
        with gr.Column():
            gr.Examples(
                examples=[
                    ["demo_images/train_2956_0001.png", "Where are the airplanes located and what is their type?"],
                    ["demo_images/7292.JPG", "How many buildings are flooded?"],
                ],
                inputs=[image, text_input],
                fn=example_trigger,
                run_on_click=True,
                outputs=[upload_flag, replace_flag],
            )
        with gr.Column():
            gr.Examples(
                examples=[
                    ["demo_images/church_183.png", "Classify the image in the following classes: Church, Beach, Dense Residential, Storage Tanks."],
                    ["demo_images/04444.png", "[identify] what is this {<8><26><22><37>}"],
                ],
                inputs=[image, text_input],
                fn=example_trigger,
                run_on_click=True,
                outputs=[upload_flag, replace_flag],
            )

    dataset.click(
        gradio_taskselect,
        inputs=[dataset],
        outputs=[text_input, task_inst],
        show_progress="hidden",
        postprocess=False,
        queue=False,
    )

    text_input.submit(
        gradio_ask,
        [text_input, chatbot, chat_state, image, img_list, upload_flag, replace_flag],
        [text_input, chatbot, chat_state, img_list, upload_flag, replace_flag], queue=False
    ).success(
        gradio_stream_answer,
        [chatbot, chat_state, img_list, temperature],
        [chatbot, chat_state]
    ).success(
        gradio_visualize,
        [chatbot, image],
        [chatbot],
        queue=False,
    )

    send.click(
        gradio_ask,
        [text_input, chatbot, chat_state, image, img_list, upload_flag, replace_flag],
        [text_input, chatbot, chat_state, img_list, upload_flag, replace_flag], queue=False
    ).success(
        gradio_stream_answer,
        [chatbot, chat_state, img_list, temperature],
        [chatbot, chat_state]
    ).success(
        gradio_visualize,
        [chatbot, image],
        [chatbot],
        queue=False,
    )

    clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, chat_state, img_list], queue=False)

demo.launch(
    share=False,
    server_name=os.getenv('GRADIO_SERVER_NAME') or '0.0.0.0',
    server_port=int(os.getenv('GRADIO_SERVER_PORT') or 8080),
)
