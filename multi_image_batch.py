import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json


def get_messages(json_file):
    # 读取 JSON 文件，注意确保文件编码与实际编码一致（例如 utf-8）
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)[39:]

    # 遍历 JSON 文件中每个字典
    for item in data:
        GT_IDs = item.get("GT_IDs")
        T_query = item.get("T_query")

        # 获取字典中对应 "ImagePoolFolder" 的路径
        folder = item.get("ImagePoolFolder")
        if folder and os.path.isdir(folder):
            # 用于存放所有图片绝对路径的列表
            abs_image_paths = []
            # 获取目录中的每个文件
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    # 获取文件的绝对路径并添加到列表中
                    abs_path = os.path.abspath(file_path)
                    abs_image_paths.append(abs_path)

        abs_image_paths.sort()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,  # 使用图片的绝对路径
                    }
                    for image_path in abs_image_paths
                ]
                + [
                    {
                        "type": "text",
                        "text": f"Please carefully understand these images and help me find the images that match the description of T_query. Only list it without explaining the reason. The content of the T_query is: {T_query}.",
                    }
                ],
            }
        ]

        yield messages, folder, GT_IDs, T_query


def main(json_file):
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
    min_pixels = 128 * 28 * 28
    max_pixels = 256 * 28 * 28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    processor.tokenizer.padding_side = "left"

    # Messages containing multiple images and a text query
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/1.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/2.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/3.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/4.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/5.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/6.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/7.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/8.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/9.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/10.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/11.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/12.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/13.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/14.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/15.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/16.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/17.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/18.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/19.png"},
    #             {"type": "image", "image": "/mnt2/lei/Qwen2.5-VL/FPS_case/ImagePoolFolder/20.png"},
    #             {
    #                 "type": "text",
    #                 "text": "Please carefully understand these images and help me find the image for first person shooter game screenshots. This type of image simulates holding a gun or other weapon from a first person perspective. Just list it without explaining the reason.",
    #             },
    #         ],
    #     }
    # ]

    with torch.no_grad():
        for messages, folder, GT_IDs, T_query in get_messages(json_file):
            try:

                print(f"Processing images in folder: {folder}")
                save_path = folder.replace("/mnt2/lei", "./dataset_multi_image")
                os.makedirs(save_path, exist_ok=True)

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

                output_content = {
                    "prediction": output_text,
                    "GT_IDs": GT_IDs,
                }

                with open(f"{save_path}/output.json", "w", encoding="utf-8") as f:
                    json.dump(output_content, f, ensure_ascii=False, indent=4)

                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error: {e}")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    json_file = "/mnt2/lei/Qwen2.5-VL/dataset/annotation.json"
    main(json_file)
