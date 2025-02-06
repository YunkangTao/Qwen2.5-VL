import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto")

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processor
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256 * 28 * 28
max_pixels = 512 * 28 * 28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# Messages containing multiple images and a text query
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/1.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/2.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/3.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/4.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/5.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/6.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/7.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/8.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/9.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/10.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/11.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/12.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/13.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/14.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/15.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/16.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/17.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/18.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/19.png"},
            {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/20.png"},
            {
                "type": "text",
                "text": "Please carefully understand these images and help me find the image for first person shooter game screenshots. This type of image simulates holding a gun or other weapon from a first person perspective. Just list it without explaining the reason.",
            },
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text)
