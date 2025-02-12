import json
import os


def compute_recall(ground_truth, prediction):
    """
    计算 recall 值
    参数:
      ground_truth (list): 真实结果列表
      prediction (list): 预测结果列表
    返回:
      float: recall 值，计算公式为 (正确预测数量 / ground_truth 数量)

    说明:
      如果 ground_truth 列表为空，返回 0.0 。
      此处采用集合交集来计算正确预测的元素。若 ground_truth 中存在重复元素，
      则需要根据实际需求做更详细的统计。
    """
    if not ground_truth:
        return 0.0

    # 将列表转换为集合以去重，然后计算交集元素
    gt_set = set(int(x) for x in ground_truth)
    pred_set = set(int(x) for x in prediction)

    # 真阳性（ground_truth 中预测正确的元素）
    true_positives = gt_set.intersection(pred_set)

    recall = len(true_positives) / len(gt_set)
    return recall


def compute_precision(ground_truth, prediction):
    if not prediction:
        return 0.0

    # 将列表转换为集合以去重，然后计算交集元素
    gt_set = set(int(x) for x in ground_truth)
    pred_set = set(int(x) for x in prediction)

    # 真阳性（ground_truth 中预测正确的元素）
    true_positives = gt_set.intersection(pred_set)

    precision = len(true_positives) / len(pred_set)
    return precision


def find_all_json_files(directory):
    """
    遍历给定目录下所有 .json 文件，并返回文件的绝对路径列表

    :param directory: 要遍历的目录路径
    :return: 包含所有 JSON 文件绝对路径的列表
    """
    json_file_paths = []

    # os.walk 会递归遍历所有子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('output.json'):
                # 拼接绝对路径并添加到列表中
                absolute_path = os.path.abspath(os.path.join(root, file))
                json_file_paths.append(absolute_path)

    return sorted(json_file_paths)


def main(json_root_path):
    all_json_files = find_all_json_files(json_root_path)
    recall_values = []
    precision_values = []
    for json_file in all_json_files:
        # 读取 JSON 文件中的数据
        with open(json_file, 'r') as f:
            data = json.load(f)

        # 计算 recall 值
        recall_value = compute_recall(data['GT_IDs'], data['prediction'])
        print("json_file:", json_file, ", Recall:", recall_value)
        recall_values.append(recall_value)

        # 计算 precision 值
        precision_value = compute_precision(data['GT_IDs'], data['prediction'])
        print("json_file:", json_file, ", Precision:", precision_value)
        precision_values.append(precision_value)

    print("Recall values:", recall_values)
    print("Number of files:", len(recall_values))
    print("1-40 average recall:", sum(recall_values[:40]) / 40)
    print("41-80 average recall:", sum(recall_values[40:80]) / 40)
    print("Average recall:", sum(recall_values) / len(recall_values))

    print("Precision values:", precision_values)
    print("Number of files:", len(precision_values))
    print("1-40 average precision:", sum(precision_values[:40]) / 40)
    print("41-80 average precision:", sum(precision_values[40:80]) / 40)
    print("Average precision:", sum(precision_values) / len(precision_values))


# 测试示例
if __name__ == "__main__":
    # ground_truth = [1, 3, 6, 12, 19]
    # prediction = [1, 3, 6, 8, 12, 19]
    # recall_value = compute_recall(ground_truth, prediction)
    # print("Recall:", recall_value)
    json_root_path = "/mnt2/lei/Qwen2.5-VL/result_one_by_one_refine_tquery_20250210"
    main(json_root_path)
