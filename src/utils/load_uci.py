import pandas as pd

def load_adult(n_sens=2):
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income"
    ]

    adult = pd.read_csv(
        "data_uci/Adult/adult.data",
        names=columns,
        sep=",",
        skipinitialspace=True
    )
    used_cols = [
        "age", 
        "fnlwgt", 
        "education_num",
        "capital_gain", 
        "hours_per_week"
    ]

    features_wo = adult[used_cols].copy() 

    race = adult["race"].copy().map({
        "White":0, "Black":1, "Asian-Pac-Islander":2, "Amer-Indian-Eskimo":3, "Other":4
    })

    sex = adult["sex"].copy().map({
        'Male':0,
        'Female':1
    })
    sens_df = pd.DataFrame({"sex":sex,"race":race})
    sens_df = sens_df.iloc[:, :n_sens]

    features_w = features_wo.copy() 
    for col in sens_df.columns:
        features_w[col] = sens_df[col]
    y_true = adult["income"].copy().map({
        '<=50K':0,
        '>50K':1
    })
    print("sens df ", sens_df)
    return features_w, features_wo, sens_df, y_true

def load_diabetes(): 
    df = pd.read_csv("data_uci/Diabetes/diabetic_data.csv")

    idx = [i for i in df.index if df.iloc[i]["gender"] != "Unknown/Invalid"] # this removes 3 / 101766 rows
    df = df.iloc[idx]

    selected = ["age", "time_in_hospital"]
    features_wo = df[selected].copy()
    age_map = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95,
    }

    features_wo["age"] = features_wo["age"].map(age_map)
    print(features_wo)
    sensitive = df["gender"].copy().map(
        {
            "Female":0, 
            "Male":1
        }
    ).rename("sex")
    sensitive = pd.DataFrame(sensitive)
    features_w = features_wo.copy() 
    features_w["sex"] = sensitive
    y_true = df["readmitted"].map({
        "NO":0, 
        ">30":1, 
        "<30":2
    })
    return features_w, features_wo, sensitive, y_true

def load_bank(): 
    selected_backurs = ["age","balance","duration"]
    df = pd.read_csv("data_uci/Bank/bank-full.csv", sep=";")

    features_wo = df.copy()[selected_backurs]
    features_w = features_wo.copy()
    sensitive = pd.DataFrame(df["marital"].map({
        'married':0, 
        'single':1,
        'divorced':1
    }))
    features_w["marital"] = sensitive

    y = df["y"].map({
        "no":0, 
        "yes":1
    })
    print(y.unique())

    return features_w, features_wo, sensitive, y

def load_census(n_sens=2):
    df = pd.read_csv("data_uci/Census/USCensus1990.data.txt")

    cols = ["dAncstry1", "dAncstry2", "iAvail", "iCitizen",
            "iClass", "dDepart", "iFertil", "iDisabl1", "iDisabl2", 
             "iEnglish", "iFeb55", "dHispanic", "dHour89" ]


    features_wo = df[cols]
    sens_df = pd.DataFrame({"sex": df["iSex"].copy(),"age": df["dAge"].copy()})
    sens_df = sens_df.iloc[:, :n_sens]
    features_w = features_wo.copy() 
    for col in sens_df.columns:
        features_w[col] = sens_df[col]

    y = df["dOccup"].copy()
    return features_w, features_wo, sens_df, y

def load_creditcard():
    df = pd.read_excel("data_uci/Creditcard/creditcard_data.xls",header=1)

    selected = [
        "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", 
        "BILL_AMT6", "LIMIT_BAL", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", 
        "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]
    features_wo = df[selected]

    sensitive = df["SEX"] - 1
    sensitive = pd.DataFrame(sensitive.rename("sex"))
    y = df['default payment next month']

    features_w = features_wo.copy() 
    features_w["sex"] = sensitive
    return features_w, features_wo, sensitive,y
import numpy as np
if __name__ == "__main__":
    for j,f in enumerate([load_adult, load_bank, load_census, load_creditcard, load_diabetes]):
        print("j ", j)
        a,b,c,d = f()

        for i,x in enumerate([a,b,c,d]):
            print(i, " has NA? ", a.isna().any().any())
            print(i, " has inf? ", np.isinf(a.to_numpy()).any() )


        distribution = (
        c.value_counts(normalize=True)
            .sort_index()
            .round(3)
                .tolist()
        )

        print("distr : ",distribution)