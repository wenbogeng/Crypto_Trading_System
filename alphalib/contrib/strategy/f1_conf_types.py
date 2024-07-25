import dataclasses
from typing import Any, Literal

CalcFactorType = Literal['cross', 'vertical']


@dataclasses.dataclass(frozen=True)  # 设置frozen=True来自动提供__hash__和__eq__方法
class F1FactorConfig:
    factor_name: str = 'FACTOR_NAME'
    descending: bool = True
    back_periods: int = 96
    fdiff_n: float = 0
    weight: float = 1


def factor_to_col_name(conf: F1FactorConfig):
    factor_name = conf.factor_name
    back_periods = conf.back_periods
    fdiff_n = conf.fdiff_n
    return f'{factor_name}_bh_{back_periods}' + (f'_diff_{fdiff_n}' if fdiff_n > 0 else '')


@dataclasses.dataclass(frozen=True)  # 设置frozen=True来自动提供__hash__和__eq__方法
class F1FilterFactorConfig:
    factor_name: str = 'FILTER_NAME'
    params: Any = 24

    def to_col_name(self) -> str:
        return f"{self.factor_name}_fl_{str(self.params)}"

    def __hash__(self):
        return hash(self.to_col_name())


@dataclasses.dataclass
class F1FilterParams:
    direction: Literal['df1', 'df2'] = 'df1'
    filter_factor: str = 'FILTER_NAME'
    filter_type: Literal['value', 'rank', 'pct'] = 'value'
    compare_operator: Literal['lt', 'gt', 'bt', 'nbt', 'lte', 'gte', 'bte', 'nbte', 'eq', 'ne'] = 'lt'
    filter_value: Any = 0.2
    rank_ascending: bool = False
    filter_after: bool = False

    def to_filter_config(self) -> F1FilterFactorConfig:
        factor_name, params = self.filter_factor.split('_fl_')
        return F1FilterFactorConfig(factor_name=factor_name, params=eval(params))
