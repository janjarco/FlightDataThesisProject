# filter df.columns.values  by regex
def filter_columns_by_regex(df, regex):
    import re
    return [col for col in df.columns.values if bool(re.search(pattern=str(regex), string=str(col)))]

import scipy.stats as stats
