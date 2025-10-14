# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2022/11/1 22:28
# @Author : caisj
# @Email : cai.sj@foxmail.com

import multiprocessing
import numpy as np
import pandas as pd



def run_task_multiprocessing(data, threads, method):
    global columns
    batch_size = int(np.ceil(len(data)/threads))
    m = multiprocessing.Manager()
    ns = m.Namespace()
    # l = multiprocessing.Lock()
    ns.data = data
    pool = multiprocessing.Pool(processes=threads)

    pool_res = []
    for batch_index in np.arange(0,data.shape[0],batch_size):
        if batch_index + batch_size >= data.shape[0]:
            index = (batch_index,)
        else:
            index = (batch_index,batch_index + batch_size)
        pool_res.append(pool.apply_async(method, args=(ns, index)))
        # r.get(timeout=1)
    pool.close()
    pool.join()

    df_res_list = []
    for result in pool_res:
        part_res = result.get()
        if len(part_res) > 1:
            columns = part_res.columns
            df_res_list.extend(np.array(part_res))
    return pd.DataFrame(df_res_list, columns=columns)