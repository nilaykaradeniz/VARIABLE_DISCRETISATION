import pandas as pd
import matplotlib.pyplot as plt
import warnings
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import ArbitraryNumberImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.imputation import EndTailImputer
from feature_engine.imputation import RandomSampleImputer
from feature_engine.imputation import AddMissingIndicator
from feature_engine.imputation import DropMissingData
import missingno as msno
from scipy.stats import kurtosis, skew
from helpers import eda

pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.filterwarnings('ignore')
pd.set_option("display.max_rows",None)
pd.set_option('display.expand_frame_repr', False)


## https://www.numpyninja.com/post/feature-engineering-handling-missing-data-with-python


df=eda.csv_file("WORK_FILE").copy()
cat_cols, num_cols, cat_but_car,typless_cols =eda.col_types(df)
na_col,null_high_col_name=eda.desc_statistics(df,num_cols,cat_cols,na_rows=True,high_null_count=True,quantile=True)


def missing_correlation(dataframe):
    question = input("Write YES if you want to apply value to certain variables. Write NO if you don't want it to apply to all variables").upper()
    if question == "YES":
        col_list = []
        while True:
            col_name = input("Please enter the variable name... Write 'STOP' for stop !!!")
            if col_name != 'STOP':
                col_list.append(col_name)
                if col_name not in dataframe.columns:
                    col_list.remove(col_name)
            else:
                print("Added variables...")
                msno.matrix(dataframe[col_list])
                msno.heatmap(dataframe[col_list])
                plt.show()
                break
    elif question == "NO":
        msno.matrix(dataframe,figsize=(10,8), fontsize=8)
        msno.heatmap(dataframe,figsize=(10,8), fontsize=8)
        plt.show()
    else:
        print("Wrong information...")
    return None


def dict_append(dataframe,dict_statistic,col,mean=False,median=False,mode=False):
    if mean:
        dict_statistic["col_name"].append(col)
        dict_statistic["value"].append(dataframe[col].mean())
        dict_statistic["statistic"].append("mean")
    if median:
        dict_statistic["col_name"].append(col)
        dict_statistic["value"].append(dataframe[col].median())
        dict_statistic["statistic"].append("median")
    if mode:
        dict_statistic["col_name"].append(col)
        dict_statistic["value"].append(dataframe[col].mode().to_string(index=False))
        dict_statistic["statistic"].append("mode")


def missing_imputation(dataframe,num_col=True,cat_col=True):
    dict_statistic = {"col_name": [], "value": [],"statistic": [] }
    if num_col:
        for col in na_col:
            if col in num_cols:
                if -1 < kurtosis(dataframe[col], nan_policy="omit") < 1 and -1 < skew(dataframe[col], nan_policy="omit") < 1:
                    dict_append(dataframe,dict_statistic,col,mean=True)
                    dataframe[col].fillna(dataframe[col].mean(), inplace=True)
                else:
                    dict_append(dataframe,dict_statistic,col,median=True)
                    dataframe[col].fillna(dataframe[col].median(), inplace=True)
    if cat_col:
        for col in na_col:
            if col in cat_cols:
                dict_append(dataframe,dict_statistic,col,mode=True)
                dataframe[col].fillna(dataframe[col].mode().to_string(index=False),inplace=True)

    fillna_df = pd.DataFrame(dict_statistic)
    print(fillna_df)

    return fillna_df


def MeanMedianImputerLibrary(dataframe):
    cols_mean = []
    cols_median = []
    for col in na_col:
        if col in num_cols:
            if -1 < kurtosis(dataframe[col], nan_policy="omit") < 1 and -1 < skew(dataframe[col], nan_policy="omit") < 1:
                cols_mean.append(col)
            else:
                cols_median.append(col)
    imputer_mean = MeanMedianImputer(imputation_method='mean', variables=cols_mean)
    dataframe = imputer_mean.fit_transform(dataframe)
    imputer_median = MeanMedianImputer(imputation_method='median', variables=cols_median)
    dataframe = imputer_median.fit_transform(dataframe)

    mean_df = pd.DataFrame([imputer_mean.imputer_dict_]).T.reset_index().set_axis(['col_name', "value"], axis=1)
    mean_df["statistic"] = imputer_mean.get_params()["imputation_method"]
    median_df = pd.DataFrame([imputer_median.imputer_dict_]).T.reset_index().set_axis(['col_name', "value"], axis=1)
    median_df["statistic"] = imputer_median.get_params()["imputation_method"]

    mean_df=pd.concat([mean_df,median_df],ignore_index=True)
    print(mean_df)
    return mean_df,dataframe


def ArbitraryNumberImputerLibrary(dataframe,constant_number=-999,constant=False,dict_constant=True):
    na_num_cols=[]
    for col in na_col:
        if col in num_cols:
            na_num_cols.append(col)
    if constant:
        question = input("Write YES if you want to apply ArbitraryNumberImputer to certain variables. Write NO if you don't want it to apply to all variables").upper()
        if question == "YES":
            col_list = []
            while True:
                col_name = input("Please enter the variable name... Write 'STOP' for stop !!!")
                if col_name != 'STOP':
                    col_list.append(col_name)
                    if col_name not in dataframe.columns:
                        col_list.remove(col_name)
                else:
                    print("Added variables...")
                    transformer = ArbitraryNumberImputer(variables=col_list, arbitrary_number=constant_number)
                    dataframe = transformer.fit_transform(dataframe)
                    break
        elif question=="NO":
            transformer = ArbitraryNumberImputer(variables=na_num_cols, arbitrary_number= constant_number)
            dataframe = transformer.fit_transform(dataframe)
        else:
            print("Wrong information...")

    if dict_constant:
        col_dict = {}
        key_list = []
        value_list = []
        while True:
            key = input("Please enter the variable name... Write 'STOP' for stop !!!")
            if key != 'STOP':
                key_list.append(key)
                if key not in dataframe.columns:
                    key_list.remove(key)
                else:
                    if key_list != []:
                        try:
                            value = float(input("Please enter the numerical value..."))
                            value_list.append(value)
                        except UnboundLocalError:
                            print("You must enter numerical value...")
            else:
                print("Added variables...")
                break
            for i in range(len(key_list)):
                col_dict[key_list[i]] = value_list[i]
            transformer = ArbitraryNumberImputer(imputer_dict =col_dict)
            dataframe = transformer.fit_transform(dataframe)

    return dataframe


def CategoricalImputerLibrary(dataframe,method_name="frequent"):
    #If you want, you can choose the "missing" method instead of "frequency".
    cols_list = []
    for col in na_col:
        if col in cat_cols:
            cols_list.append(col)
    imputer= CategoricalImputer(imputation_method=method_name, variables=cols_list)
    dataframe = imputer.fit_transform(dataframe)

    cat_df = pd.DataFrame([imputer.imputer_dict_]).T.reset_index().set_axis(['col_name', "value"], axis=1)
    cat_df["statistic"] = imputer.get_params()["imputation_method"]
    print(cat_df)
    return cat_df,dataframe


def EndTailImputerLibrary(dataframe,method="gaussian",tail="right",fold=1.5):

        # """
        #     Gaussian limits:
        # right tail: mean + 3*std
        # left tail: mean - 3*std
        #
        #     IQR limits:
        # right tail: 75th quantile + 3*IQR
        # left tail: 25th quantile - 3*IQR
        # where IQR is the inter-quartile range = 75th quantile - 25th quantile
        #
        #     Maximum value:
        # right tail: max * 3
        # left tail: not applicable
        # """

    cols = []
    for col in na_col:
        if col in num_cols:
            cols.append(col)
    question = input("Write YES if you want to apply EndTailImputer to certain variables. Write NO if you don't want it to apply to all variables").upper()
    if question == "YES":
        col_list = []
        while True:
            col_name = input("Please enter the variable name... Write 'STOP' for stop !!!")
            if col_name != 'STOP':
                col_list.append(col_name)
                if col_name not in dataframe.columns:
                    col_list.remove(col_name)
            else:
                print("Added variables...")
                imputer_tail = EndTailImputer(imputation_method=method, tail=tail,fold=fold,variables=col_list)
                dataframe = imputer_tail.fit_transform(dataframe)
                print("EndTailImputer applied to the variables you wrote. --> imputation_method :", method, ", tail :", tail,", fold :", fold)
                break
    elif question == "NO":
        imputer_tail = EndTailImputer(imputation_method=method, tail=tail,fold=fold,variables=cols)
        dataframe = imputer_tail.fit_transform(dataframe)
        print("EndTailImputer applied to all variables. --> imputation_method :" ,method,", tail :", tail,", fold :" ,fold )
    else:
        print("Wrong information...")

    tail_df = pd.DataFrame([imputer_tail.imputer_dict_]).T.reset_index().set_axis(['col_name', "value"], axis=1)
    tail_df["statistic"] = imputer_tail.get_params()["imputation_method"]
    print(tail_df)
    return tail_df,dataframe


def RandomSampleImputerLibrary(dataframe,seed="observation",seeding_method='add'):
    cols = []
    for col in na_col:
        if col in num_cols:
            cols.append(col)
    question = input("Write YES if you want to apply RandomSampleImputer to certain variables. Write NO if you don't want it to apply to all variables").upper()
    if question == "YES":
        col_list = []
        while True:
            col_name = input("Please enter the variable name... Write 'STOP' for stop !!!")
            if col_name != 'STOP':
                col_list.append(col_name)
                if col_name not in dataframe.columns:
                    col_list.remove(col_name)
            else:
                print("Added variables...")
                imputer_random = RandomSampleImputer(random_state=col_list,seed=seed,seeding_method=seeding_method)
                dataframe = imputer_random.fit_transform(dataframe)
                print("RandomSampleImputer applied to the variables you wrote.")
                break
    elif question == "NO":
        imputer_random = RandomSampleImputer(random_state=cols,seed=seed,seeding_method=seeding_method)
        dataframe = imputer_random.fit_transform(dataframe)
        print("RandomSampleImputer applied to all variables.")
    else:
        print("Wrong information...")
    return dataframe


def AddMissingIndicatorLibrary(dataframe):
    question = input("Write YES if you want to apply AddMissingIndicator to certain variables. Write NO if you don't want it to apply to all variables").upper()
    if question == "YES":
        col_list = []
        while True:
            col_name = input("Please enter the variable name... Write 'STOP' for stop !!!")
            if col_name != 'STOP':
                col_list.append(col_name)
                if col_name not in dataframe.columns:
                    col_list.remove(col_name)
            else:
                print("Added variables...")
                imputer_na_binary = AddMissingIndicator(variables=col_list)
                dataframe = imputer_na_binary.fit_transform(dataframe)
                print("AddMissingIndicator applied to the variables you wrote.")
                break
    elif question == "NO":
        excepted_varibles=list(dataframe.T[dataframe.nunique()==2].index)
        imputer_na_binary = AddMissingIndicator( variables=list(dataframe.columns.drop(excepted_varibles)))
        dataframe = imputer_na_binary.fit_transform(dataframe)
        print("AddMissingIndicator applied to all variables. Excepted this variables --> ",excepted_varibles)
    else:
        print("Wrong information...")
    binary_na_columns= pd.DataFrame(list(dataframe.T[dataframe.columns.str.contains("na")].index),columns=["col_name"])
    print("Binary variables added to the dataset","\n", binary_na_columns)
    return  binary_na_columns,dataframe


def DropMissingDataLibrary(dataframe,null_high_col,null_high=False):
    original_index = list(dataframe.index)
    if null_high:
        imputer_drop = DropMissingData(variables=null_high_col)
        dataframe = imputer_drop.fit_transform(dataframe)
    question = input("Write YES if you want to apply DropMissingData to certain variables. Write NO if you don't want it to apply to all variables").upper()
    if question == "YES":
        col_list = []
        while True:
            col_name = input("Please enter the variable name... Write 'STOP' for stop !!!")
            if col_name != 'STOP':
                col_list.append(col_name)
                if col_name not in dataframe.columns:
                    col_list.remove(col_name)
            else:
                print("Added variables...")
                imputer_drop = DropMissingData(variables=col_list)
                dataframe = imputer_drop.fit_transform(dataframe)
                print("DropMissingData applied to the variables you wrote.")
                break
    elif question == "NO":
        imputer_drop = DropMissingData( variables=list(dataframe.columns))
        dataframe = imputer_drop.fit_transform(dataframe)
        print("DropMissingData applied to all variables.")
    else:
        print("Wrong information...")

    after_drop_index=list(dataframe.index)
    delete_index = [i for i in original_index if i not in after_drop_index]
    print("Deleted indexes total : ",len(delete_index),"\n", delete_index)
    return dataframe


