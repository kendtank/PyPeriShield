#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 14:44
@Description: demo - manage的一个demo实现
@Modify:
@Contact: tankang0722@gmail.com
"""
import yaml

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data



if __name__ == '__main__':
    data = load_yaml_file('/home/lyh/work/depoly/PyPeriShield-feature/config/mq_parameters.yaml')
    print(data, type(data))
