from config import *

def parse(opt):
    """解析命令行参数

    Args:
        opt (_type_): 命令行参数

    Returns:
        tuple: data_config, model_config - 划分使用的配置（voc 或 pest）
    """
    # 定义数据配置映射字典
    data_config_map = {
        'voc12': voc12_config,
        'voc07': voc07_config,
        'pestv3': pestv3_config,
        'coco': coco_config,
        'visdrone': visdrone_config,
        'pestv1': pestv1_config,
        'ip102': ip102_config,
        'pest24': pest24_config
    }

    # 根据 opt.dataset 获取对应的数据配置
    if opt.dataset in data_config_map:
        data_config = data_config_map[opt.dataset]
    else:
        raise ValueError(f"Invalid dataset: {opt.dataset}. Available options are {', '.join(data_config_map.keys())}.")

    # 定义模型配置映射字典
    model_config_map = {
        'voc12': judge_config['voc12'],
        'voc07': judge_config['voc07'],
        'pestv3': judge_config['pestv3'],
        'coco': judge_config['coco'],
        'visdrone': judge_config['visdrone'],
        'pestv1': judge_config['pestv1'],
        'ip102': judge_config['ip102'],
        'pest24': judge_config['pest24']
    }

    # 根据 opt.model 获取对应的模型配置
    if opt.model_zoo in model_config_map:
        model_config = model_config_map[opt.model_zoo]
    else:
        raise ValueError(f"Invalid model: {opt.model}. Available options are {', '.join(model_config_map.keys())}.")

    return data_config, model_config

def get_model(mconfig):
    """获取用于judge的模型

    Args:
        mconfig (dict): config.py['which']
    """
    model_list = []
    for model in mconfig['models']:
        model_list.append(YOLO(model))
    threshold = mconfig['threshold']
    return threshold, model_list