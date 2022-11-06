# import required libraries

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#!pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

# some settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
############################################################################################################################################
############################################################################################################################################
# Features

# Impression: Number of ad views
# Click: The number of clicks on the displayed ad
# Purchase: The number of products purchased after the ads clicked
# Earning: Earnings after purchased products
############################################################################################################################################
############################################################################################################################################
# read the datasets
control = pd.read_excel("ab_testing_veri220805073728-221026-192135/ab_testing.xlsx", sheet_name="Control Group")
test = pd.read_excel("ab_testing_veri220805073728-221026-192135/ab_testing.xlsx", sheet_name="Test Group")
df_control = control.copy()
df_test = test.copy()

# examine the control and test datasets
df_control.head(5)
"""
    Impression      Click  Purchase    Earning
0  82529.45927 6090.07732 665.21125 2311.27714
1  98050.45193 3382.86179 315.08489 1742.80686
2  82696.02355 4167.96575 458.08374 1797.82745
3 109914.40040 4910.88224 487.09077 1696.22918
4 108457.76263 5987.65581 441.03405 1543.72018
"""
df_test.head(5)
"""
    Impression      Click  Purchase    Earning
0 120103.50380 3216.54796 702.16035 1939.61124
1 134775.94336 3635.08242 834.05429 2929.40582
2 107806.62079 3057.14356 422.93426 2526.24488
3 116445.27553 4650.47391 429.03353 2281.42857
4 145082.51684 5201.38772 749.86044 2781.69752
"""
# assign the letter "C" to the variable names of the control group and "T" to the variable names of the test group
df_control.columns = ["C_" + i for i in df_control.columns]
df_test.columns = ["T_" + i for i in df_test.columns]

############################################################################################################################################
# overview of the control and test datasets
def check_df(dataframe, head=5):
    print("##################### SHAPE #####################")
    print(dataframe.shape)
    print("##################### INFO #####################")
    print(dataframe.info())
    print("##################### DESCRIPTION #####################")
    print(dataframe.describe().T)
    print("##################### HEAD #####################")
    print(dataframe.head(head))
    print("##################### MISSING VALUE #####################")
    print(dataframe.isnull().sum())
    print("##################### DUPLICATE VALUE #####################")
    print(dataframe.duplicated().sum())

check_df(df_control)
"""
##################### DESCRIPTION #####################
                count         mean         std         min         25%         50%          75%          max
C_Impression 40.00000 101711.44907 20302.15786 45475.94296 85726.69035 99790.70108 115212.81654 147539.33633
C_Click      40.00000   5100.65737  1329.98550  2189.75316  4124.30413  5001.22060   5923.80360   7959.12507
C_Purchase   40.00000    550.89406   134.10820   267.02894   470.09553   531.20631    637.95709    801.79502
C_Earning    40.00000   1908.56830   302.91778  1253.98952  1685.84720  1975.16052   2119.80278   2497.29522
"""
check_df(df_test)
"""
##################### DESCRIPTION #####################
                count         mean         std         min          25%          50%          75%          max
T_Impression 40.00000 120512.41176 18807.44871 79033.83492 112691.97077 119291.30077 132050.57893 158605.92048
T_Click      40.00000   3967.54976   923.09507  1836.62986   3376.81902   3931.35980   4660.49791   6019.69508
T_Purchase   40.00000    582.10610   161.15251   311.62952    444.62683    551.35573    699.86236    889.91046
T_Earning    40.00000   2514.89073   282.73085  1939.61124   2280.53743   2544.66611   2761.54540   3171.48971
"""
# According to the method offered, the purchase number of the test group is higher.
# While the number of ads viewed increases, the number of clicks decreases.
# Despite the decrease in the number of clicks, the income has increased.

############################################################################################################################################
# With the concat method, combine the control and test group data after the analysis process.
df = pd.concat([df_control, df_test], axis=1)
df.head(5)

############################################################################################################################################
############################################################################################################################################
# Let's start A/B Testing

# 1. Define the Hypothesis
# 2. Check the Assumptions
#   - 1. Normality Assumption (+)
#   - 2. Variance Homogeneity Assumption (+)
# 3. Apply the hypothesis
#   - 1. If the assumptions are provided, apply the independent two-sample t-test (Parametric Test) (+)
#   - 2. If the assumptions are not provided, apply the Mann-Whitney U Test (Non-Parametric Test)

# NOTE: If the normality assumption is not provided, the Mann-Whitney U Test is applied.
#       If the normality assumption is provided but the variance homogeneity assumption is not provided,
#       the equal_var parameter can be set to False in the independent two-sample t-test.

# (+) indicates that the method is applied.

# The purpose here is to test whether there is a statistically significant difference between the Purchase numbers.

# Define the hypothesis
# H0: M1 = M2 (There is no statistically significant difference between the number of purchases.)
# H1: M1!= M2 (There is a statistically significant difference between the number of purchases.)

# Analyze the average purchase number of the control and test groups
df.C_Purchase.mean() # 550.8940587702316
df.T_Purchase.mean() # 582.1060966484677
# According to the results, the average purchase number of the test group is higher than the control group.
# But is this difference statistically significant?

############################################################################################################################################
############################################################################################################################################

# Assumption Test 1: Normality Assumption
# H0: The assumption of normal distribution is provided.
# H1: The assumption of normal distribution is not provided.
# p < 0.05 H0 REJECTED, p > 0.05 H0 CANNOT BE REJECTED

test_stat, pvalue = shapiro(df.C_Purchase)
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue)) # Test Stat = 0.9773, p-value = 0.5891
# pvalue > 0.05, H0 cannot be rejected. The assumption of normal distribution is provided.

test_stat, pvalue = shapiro(df.T_Purchase)
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue)) # Test Stat = 0.9589, p-value = 0.1541
# pvalue > 0.05, H0 cannot be rejected. The assumption of normal distribution is provided.


# Assumption Test 2: Homogeneity of Variance
# H0: The assumption of homogeneity of variance is provided.
# H1: The assumption of homogeneity of variance is not provided.
# p < 0.05 H0 REJECTED, p > 0.05 H0 CANNOT BE REJECTED

test_stat, pvalue = levene(df.C_Purchase, df.T_Purchase)
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue)) # Test Stat = 2.6393, p-value = 0.1083
# pvalue > 0.05, H0 cannot be rejected. The assumption of homogeneity of variance is provided.

############################################################################################################################################
# Normality assumption and homogeneity of variance assumption are provided, so we apply the independent two-sample t-test.

# ttest if the normality assumption is provided for 2 groups, it is used.
# ttest if the normality assumption is provided and the variance homogeneity is provided, it is used.
# ttest if the normality assumption is provided and the variance homogeneity is not provided, it is used.

test_stat, pvalue = ttest_ind(df.C_Purchase, df.T_Purchase, equal_var=True) # If the assumption of homogeneity of variance is provided, equal_var=True
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue)) # Test Stat = -0.9416, p-value = 0.3493
# pvalue > 0.05, H0 cannot be rejected. There is no statistically significant difference between the number of purchases.



















