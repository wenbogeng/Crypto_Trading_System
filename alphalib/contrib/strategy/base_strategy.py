import abc
from typing import Dict, List

import pandas as pd


class BaseStrategy(abc.ABC):
    @abc.abstractmethod
    def calc_alloc_ratio(self, all_df: Dict[str, pd.DataFrame], start_date: str, end_date: str) -> pd.Series:
        raise NotImplementedError()


class MultiStrategy(BaseStrategy):
    stg_list: List[BaseStrategy]

    def __init__(self, stg_list: List[BaseStrategy]):
        self.stg_list = stg_list

    def calc_alloc_ratio(self, all_df: Dict[str, pd.DataFrame], start_date: str, end_date: str) -> pd.Series:
        res_list = [
            stg.calc_alloc_ratio(all_df=all_df, start_date=start_date, end_date=end_date)
            for stg in self.stg_list
        ]
        res = pd.concat(res_list, axis=1).sum(axis=1) / len(self.stg_list)
        return res
