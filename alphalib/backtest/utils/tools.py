from typing import Dict, List

import pandas as pd


def convert_to_filter_cls_quick(filter_list: List[str]) -> Dict[str, List[str]]:
    filter_class_list = list(set([x.split('_fl_')[0] for x in filter_list]))
    filter_cols_dic = {k: [x for x in filter_list if x.split('_fl_')[0] == k] for k in filter_class_list}
    return filter_cols_dic


def convert_to_cls(factor_list):
    cls_list = set()
    for factor_name, if_reverse, back_hour, d_num, weight in factor_list:
        cls_list.add(factor_name)

    return list(cls_list)


def convert_to_feature(factor_list):
    feature_list = set()
    for factor_name, if_reverse, back_hour, d_num, weight in factor_list:
        if d_num == 0:
            feature_list.add(f'{factor_name}_bh_{back_hour}')
        else:
            feature_list.add(f'{factor_name}_bh_{back_hour}_diff_{d_num}')

    return list(feature_list)


# 纵截面
def cal_factor_by_vertical(df, factor_list, factor_tag='因子'):
    feature_list = []
    coef_ = []
    for factor_name, if_reverse, back_hour, d_num, weight in factor_list:
        reverse_ = -1 if if_reverse else 1
        if d_num == 0:
            _factor = f'{factor_name}_bh_{back_hour}'
        else:
            _factor = f'{factor_name}_bh_{back_hour}_diff_{d_num}'
        feature_list.append(_factor)
        coef_.append(weight * reverse_)
    coef_ = pd.Series(coef_, index=feature_list)
    df[f'{factor_tag}'] = df[feature_list].dot(coef_.T)
    return df
