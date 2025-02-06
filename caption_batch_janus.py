import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json


def get_messages(json_file):
    # 读取 JSON 文件，注意确保文件编码与实际编码一致（例如 utf-8）
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 用于存放所有图片绝对路径的列表
    abs_image_paths = []

    # 遍历 JSON 文件中每个字典
    for item in data:
        GT_IDs = item.get("GT_IDs")
        T_query = item.get("T_query")

        # 获取字典中对应 "ImagePoolFolder" 的路径
        folder = item.get("ImagePoolFolder")
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
                            "text": "Describe this image.",
                        },
                    ],
                }
            ]
            messages_list.append(message)

            caseid = item.get("CaseID")

        yield messages_list, folder, GT_IDs, T_query, caseid


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
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    processor.tokenizer.padding_side = "left"

    # # Sample messages for batch inference
    # messages1 = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": "file:///path/to/image1.jpg"},
    #             {"type": "image", "image": "file:///path/to/image2.jpg"},
    #             {"type": "text", "text": "What are the common elements in these pictures?"},
    #         ],
    #     }
    # ]
    # messages2 = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Who are you?"},
    # ]
    # # Combine messages for batch processing
    # messages = [messages1, messages2]
    janus_out_path = '/mnt2/lei/Janus/output'
    with torch.no_grad():
        for messages, folder, gt_list, T_query, caseid in get_messages(json_file):
            print(f"Processing images in folder: {folder}")

            save_path = folder.replace("/mnt2/lei", "./dataset")
            os.makedirs(save_path, exist_ok=True)

            # Preparation for batch inference
            # texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            # image_inputs, video_inputs = process_vision_info(messages)
            # inputs = processor(
            #     text=texts,
            #     images=image_inputs,
            #     videos=video_inputs,
            #     padding=True,
            #     return_tensors="pt",
            # )
            # inputs = inputs.to("cuda")

            # Batch Inference
            # generated_ids = model.generate(**inputs, max_new_tokens=256)
            # generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            # output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # print(output_texts)
            janus_out = os.path.join(janus_out_path, caseid+'.json')
            with open(janus_out, "r", encoding="utf-8") as file:
                data = json.load(file)
            captions = data['captions']
            query = data['query']
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": captions[i]['answer'],
                        }
                        for i in range(len(captions))
                    ]
                    + [
                        {
                            "type": "text",
                            "text": f"The above are some descriptions of the images. Please help me find which ones are most suitable for Tquery. Just list them without explanation. For example, you can output [1, 3, 5]. The Tquery is:{query}",
                        }
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            output_content = {
                "prediction": output_text,
                "GT_IDs": gt_list,
                "res": [{"id": f"{i+1}", "answer": f"{captions[i]['answer']}", "similarity": None} for i in range(len(captions))],
            }

            with open(f"{save_path}/output.json", "w", encoding="utf-8") as f:
                json.dump(output_content, f, ensure_ascii=False, indent=4)

            torch.cuda.empty_cache()


if __name__ == "__main__":

    # 假设 JSON 文件的路径为 data.json
    json_file = "/mnt2/lei/Qwen2.5-VL/dataset/annotation.json"

    main(json_file)
