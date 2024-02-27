import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.collections import PatchCollection
from PIL import Image
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import math
import cv2
def scale_bounding_box(box, scale_factor=1.2):
    # Extracting coordinates
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = box

    # Calculating new width and height
    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y

    # Scaling width and height
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Calculating new coordinates
    new_top_left_x = top_left_x - int((new_width - width) / 2)
    new_top_left_y = top_left_y - int((new_height - height) / 2)
    new_bottom_right_x = new_top_left_x + new_width
    new_bottom_right_y = new_top_left_y + new_height

    # Returning the scaled bounding box
    return [new_top_left_x, new_top_left_y, new_bottom_right_x, new_bottom_right_y]



def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image
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


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(args.image_file)
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, args)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        import pdb;pdb.set_trace()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        # bboxes=[]
        # print(inp)
        # # if ('[refer]') or ("[grounding]") in inp:
        # output=outputs.replace('</s>','')
        # print(output)
        # # import pdb;pdb.set_trace()
        # bboxes = np.array([int(x) for y in output.replace("|", "").split("}") for x in y.replace("><", ",").replace(">", "").replace("<", "").replace("}", "").replace("{", "").split(',') if x !=""]).astype(np.float32)
        # remainder = len(bboxes)%5
        # if remainder >0:
        #     bboxes = bboxes[:-remainder]
        # # bboxes[1]=100-bboxes[1]
        # # bboxes=bboxes
        # # scaled_bbox=scale_bounding_box(bboxes[:-1], scale_factor=1.3)
        # # bboxes=scaled_bbox.append(bboxes[-1])
        # # bboxes = bboxes.reshape(-1, 5)
        # bboxes=bboxes.tolist()
        # bboxes=[int(bbox*5.04) for bbox in bboxes]
        # bboxes = np.array([bbox_and_angle_to_polygon(bboxes[0],bboxes[1],bboxes[2],bboxes[3],bboxes[4])])
        # # print(bboxes)
        # bboxes=bboxes.reshape(4,2)
        
        # image = cv2.imread(args.image_file)
        # # print(image.shape)
        # image=cv2.resize(image,(504,504))
        # plt.imshow(image)
        # # import pdb;pdb.set_trace()
        # polygons=[Polygon(bboxes)]
        # plt.axis('off')
        # ax = plt.gca()
        # ax.set_autoscale_on(False)   
        # p = PatchCollection(polygons, facecolors='none', edgecolors=[(0,255,0)], linewidths=2)
        # ax.add_collection(p)
        # # print('hello')
        # plt.savefig('/share/data/drive_3/kartik/LLaVA/output_images/output.jpg')
                
        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)
