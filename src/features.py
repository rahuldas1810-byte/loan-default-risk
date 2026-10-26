def add_features(d):
    d = d.copy()
    d["ANNUITY_INCOME_RATIO"] = d["AMT_ANNUITY"] / d["AMT_INCOME_TOTAL"]
    d["CREDIT_TERM"] = d["AMT_ANNUITY"] / d["AMT_CREDIT"]
    d["GOODS_CREDIT_RATIO"] = d["AMT_GOODS_PRICE"] / d["AMT_CREDIT"]
    d["EMPLOYED_AGE_RATIO"] = d["DAYS_EMPLOYED"] / d["DAYS_BIRTH"]
    d["INCOME_PER_PERSON"] = d["AMT_INCOME_TOTAL"] / d["CNT_FAM_MEMBERS"]
    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    d["EXT_MEAN"] = d[ext].mean(axis=1)
    d["EXT_MIN"] = d[ext].min(axis=1)
    return d