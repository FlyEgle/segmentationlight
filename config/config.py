"""make argparse
@author: FlyEgle
@datetime: 2022-01-20
"""
import yaml 
import argparse
from dotmap import DotMap


def build_argparse():
    parser = argparse.ArgumentParser()
    # -------------------------------
    parser.add_argument('--hyp', type=str, default="/data/jiangmingchao/data/code/SegmentationLight/hyparam/base.yaml")
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int)
    hyp = parser.parse_args()
    return hyp

def parse_yaml(yam_path):
    with open(yam_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    data = DotMap(data)
    return data


if __name__ == '__main__':
    yaml_file = "/data/jiangmingchao/data/code/SegmentationLight/hyparam/base.yaml"
    with open(yaml_file) as file:
        data = yaml.load(file)
    config = DotMap(data)
    print(config)
    # print(data.keys())   