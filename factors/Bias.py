def signal(*args):
    # Bias
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    ma = df['close'].rolling(n, min_periods=1).mean()
    df[factor_name] = (df['close'] / ma - 1)

    return df
