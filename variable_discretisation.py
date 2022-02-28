import pandas as pd
import numpy as np
import seaborn as sns
from helpers import eda
from helpers import missing_data_imputation
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.discretisation import EqualWidthDiscretiser
from feature_engine.discretisation import ArbitraryDiscretiser
from feature_engine.discretisation import DecisionTreeDiscretiser
pd.set_option('display.expand_frame_repr', False)


df=sns.load_dataset("titanic")
cat_cols, num_cols, cat_but_car,typless_cols =eda.col_types(df,car_th_cat_th_lower=5)
na_col,null_high_col_name=eda.desc_statistics(df,num_cols,cat_cols,na_rows=True,high_null_count=True,quantile=True)
df=missing_data_imputation.RandomSampleImputerLibrary(df)


def EqualFrequencyDiscretiserLibrary(dataframe,num_cols,q1=5,q2=4,delete_orj_column=False,plot=False):
    transformer = EqualFrequencyDiscretiser()
    dataframe2=pd.DataFrame()
    question = input("Write YES if you want to apply value to certain variables. Write NO if you don't want it to apply to all variables").upper()
    if question == "YES":
        col_list = []
        while True:
            col_name = input("Please enter the variable name... Write 'STOP' for stop !!!")
            if col_name != 'STOP':
                if col_name not in col_list:
                    col_list.append(col_name)
                    if col_name not in num_cols:
                        col_list.remove(col_name)
            else:
                if col_list != []:
                    print("Added variables...")
                    transformer = EqualFrequencyDiscretiser(q=q1, variables=col_list)
                    dataframe2 = transformer.fit_transform(dataframe[col_list]).reset_index()
                    break
    elif question == "NO":
        transformer = EqualFrequencyDiscretiser(q=q2, variables=num_cols)
        dataframe2 = transformer.fit_transform(dataframe[num_cols]).reset_index()
        print("Applied to all numeric variables..")
    else:
        print("Wrong information...")

    binning_col = ["index"]
    for col in dataframe2.columns:
        if col != "index":
            col += "_Binning"
            binning_col.append(col)
    dataframe2.set_axis(binning_col, axis=1, inplace=True)
    dataframe=dataframe.reset_index().merge(dataframe2, how="inner", on="index")
    del dataframe['index']

    bounder_df = pd.DataFrame(transformer.__dict__["binner_dict_"])
    print("\n","Category Boundaries of Variables")
    print(bounder_df)

    if delete_orj_column:
        for i in dataframe.columns:
            if (len(list(dataframe.loc[:, dataframe.columns.str.contains(i)].columns)) > 1):
                del dataframe[i]

    if plot:
        for col in dataframe.loc[:, dataframe.columns.str.contains("Binning")].columns:
            dataframe.groupby(col)[col].count().plot.bar()

    return dataframe
#df=EqualFrequencyDiscretiserLibrary(df,num_cols)



def EqualWidthDiscretiserLibrary(dataframe,num_cols,bins1=5,bins2=4,delete_orj_column=False,plot=False):
    transformer = EqualWidthDiscretiser()
    dataframe2=pd.DataFrame()
    question = input("Write YES if you want to apply value to certain variables. Write NO if you don't want it to apply to all variables").upper()
    if question == "YES":
        col_list = []
        while True:
            col_name = input("Please enter the variable name... Write 'STOP' for stop !!!")
            if col_name != 'STOP':
                if col_name not in col_list:
                    col_list.append(col_name)
                    if col_name not in dataframe.columns:
                        col_list.remove(col_name)
            else:
                if col_list != []:
                    print("Added variables...")
                    transformer = EqualWidthDiscretiser(bins=bins1, variables=col_list)
                    dataframe2 = transformer.fit_transform(dataframe[col_list]).reset_index()
                    break
    elif question == "NO":
        transformer = EqualWidthDiscretiser(bins=bins2, variables=num_cols)
        dataframe2 = transformer.fit_transform(dataframe[num_cols]).reset_index()
        print("Applied to all numeric variables..")
    else:
        print("Wrong information...")

    binning_col = ["index"]
    for col in dataframe2.columns:
        if col != "index":
            col += "_Binning"
            binning_col.append(col)
    dataframe2.set_axis(binning_col, axis=1, inplace=True)
    dataframe=dataframe.reset_index().merge(dataframe2, how="inner", on="index")
    del dataframe['index']

    bounder_df = pd.DataFrame(transformer.__dict__["binner_dict_"])
    print("\n","Category Boundaries of Variables")
    print(bounder_df)

    if delete_orj_column:
        for i in dataframe.columns:
            if (len(list(dataframe.loc[:, dataframe.columns.str.contains(i)].columns)) > 1):
                del dataframe[i]
    if plot:
        for col in dataframe.loc[:, dataframe.columns.str.contains("Binning")].columns:
            dataframe.groupby(col)[col].count().plot.bar()

    return dataframe
#df=EqualWidthDiscretiserLibrary(df,num_cols)



def ArbitraryDiscretiserLibrary(dataframe,num_cols,delete_orj_column=False,plot=True,inf_add=True):
    user_dict = {}
    key_list = []
    counter = 0
    while True:
        key = input("Please enter the variable name... Write 'STOP' for stop !!!")
        if key != 'STOP':
            key_list.append(key)
            if key not in num_cols:
                key_list.remove(key)
            else:
                if key_list != []:
                    binning_list = []
                    while True:
                        value = input("Please enter the numerical value... Write 'STOP' for stop !!!")
                        try:
                            if value != 'STOP':
                                if value not in binning_list:
                                    binning_list.append(int(value))
                                    binning_list.sort()
                            else:
                                print("Added values...")
                                user_dict[key_list[counter]] = binning_list
                                break
                        except ValueError:
                            print("You must enter numerical value...")
                    if inf_add:
                        binning_list.insert(0, -np.inf)
                        binning_list.append(np.inf)
                counter += 1
        else:
            if key_list == []:
                print("No variables in the list...")
            else:
                print("Added variables...")
            break

    transformer = ArbitraryDiscretiser(binning_dict=user_dict, return_object=False, return_boundaries=False)
    dataframe2 = transformer.fit_transform(dataframe[list(user_dict.keys())]).reset_index()

    binning_col = ["index"]
    for col in dataframe2.columns:
        if col != "index":
            col += "_Binning"
            binning_col.append(col)
    dataframe2.set_axis(binning_col, axis=1, inplace=True)
    dataframe=dataframe.reset_index().merge(dataframe2, how="inner", on="index")
    del dataframe['index']

    bounder_df = pd.DataFrame.from_dict(transformer.binner_dict_,orient='index').transpose()
    print("\n","Category Boundaries of Variables")
    print(bounder_df)

    if delete_orj_column:
        for i in dataframe.columns:
            if (len(list(dataframe.loc[:, dataframe.columns.str.contains(i)].columns)) > 1):
                del dataframe[i]

    if plot:
        for col in dataframe.loc[:, dataframe.columns.str.contains("Binning")].columns:
            dataframe.groupby(col)[col].count().plot.bar()

    return dataframe
#df=ArbitraryDiscretiserLibrary(df,num_cols)



def DecisionTreeDiscretiserLibrary(dataframe,num_cols,delete_orj_column=False,plot=False):
    cv_value = int(input("Please, enter the cross validation value..."))
    dataframe2=pd.DataFrame()
    target_col = input("Please enter the target variable name.")

    question = input("Write YES if you want to apply value to certain variables. Write NO if you don't want it to apply to all numeric variables").upper()
    if question == "YES":
        col_list = []
        while True:
            col_name = input("Please enter the variable name... Write 'STOP' for stop !!!")
            if col_name != 'STOP':
                if col_name not in col_list:
                    col_list.append(col_name)
                    if col_name not in num_cols:
                        col_list.remove(col_name)
                        if col_name==target_col:
                            col_list.remove(col_name)
            else:
                if col_list != []:
                    print("Added variables...")
                    transformer = DecisionTreeDiscretiser(cv=cv_value,
                                   scoring='neg_mean_squared_error',
                                   variables=col_list,
                                   regression=True,
                                   random_state=29)
                    dataframe2 = transformer.fit_transform(dataframe[col_list],dataframe[target_col]).reset_index()
                    break
    elif question == "NO":
        transformer =  DecisionTreeDiscretiser(cv=cv_value,
                           scoring='neg_mean_squared_error',
                           variables=num_cols,
                           regression=True,
                           random_state=29)
        dataframe2 = transformer.fit_transform(dataframe[num_cols],dataframe[target_col]).reset_index()
        print("Applied to all numeric variables..")
    else:
        print("Wrong information...")

    binning_col = ["index"]
    for col in dataframe2.columns:
        if col != "index":
            col += "_Binning"
            binning_col.append(col)
    dataframe2.set_axis(binning_col, axis=1, inplace=True)
    dataframe=dataframe.reset_index().merge(dataframe2, how="inner", on="index")
    del dataframe['index']
    del dataframe2['index']

    binning_dict={}
    for col in dataframe2.columns:
        binning_dict[col]=dataframe[col].unique()
    bounder_df = pd.DataFrame.from_dict(binning_dict, orient='index').transpose()
    print("\n","Category Boundaries of Variables")
    print(bounder_df)


    if delete_orj_column:
        for i in dataframe.columns:
            if (len(list(dataframe.loc[:, dataframe.columns.str.contains(i)].columns)) > 1):
                del dataframe[i]

    if plot:
        for col in dataframe.loc[:, dataframe.columns.str.contains("Binning")].columns:
            dataframe.groupby(col)[col].count().plot.bar()

    return dataframe
#df=DecisionTreeDiscretiserLibrary(df,num_cols)















