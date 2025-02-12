import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json


def get_messages(data_info):

    GT_IDs = data_info.get("GT_IDs")
    T_query = data_info.get("T_query")

    # 用于存放所有图片绝对路径的列表
    abs_image_paths = []

    # 获取字典中对应 "ImagePoolFolder" 的路径
    folder = data_info.get("ImagePoolFolder")
    if folder and os.path.isdir(folder):
        # 获取目录中的每个文件
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                # 获取文件的绝对路径并添加到列表中
                abs_path = os.path.abspath(file_path)
                abs_image_paths.append(abs_path)

    abs_image_paths.sort()

    messages_list = []

    for image_path in abs_image_paths:
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,  # 使用图片的绝对路径
                    },
                    {
                        "type": "text",
                        "text": f"Does this image match the description? If so, answer 1. If not, answer 0: {T_query}",
                    },
                ],
            }
        ]
        messages_list.append(message)

    return messages_list, folder, GT_IDs, T_query


def main(data_info):
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
    processor.tokenizer.padding_side = "left"

    with torch.no_grad():
        messages, folder, gt_list, T_query = get_messages(data_info)

        print(f"Processing images in folder: {folder}")

        save_path = folder.replace("/mnt2/lei", "./dataset_one_by_one")
        os.makedirs(save_path, exist_ok=True)

        # Preparation for batch inference
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Batch Inference
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(output_texts)
        # print(len(output_texts))

        # prediction_index = [f"{index:05d}" for index, value in enumerate(output_texts, start=1) if value == 1 or value == '1']

        # output_content = {"prediction_one_hot": output_texts, "prediction": prediction_index, "GT_IDs": gt_list}

        # with open(f"{save_path}/output.json", "w", encoding="utf-8") as f:
        #     json.dump(output_content, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # json_file = "/mnt2/lei/Qwen2.5-VL/dataset/annotation.json"
    data_info = {
        "CaseID": "case00044",
        "T_query": "document photos",
        "T_query_cn": "",
        "ImagePoolFolder": "/mnt2/lei/RetrievalVLM/20250107_find_five_cases/case4_find_document_photo/ImagePoolFolder/",
        "GT_IDs": ["1", "5", "8", "13", "14"],
    }
    main(data_info)
