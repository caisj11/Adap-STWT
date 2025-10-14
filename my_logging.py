# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2022/10/19 21:48
# @Author : caisj
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
fh = logging.FileHandler(r"./log.txt")
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

