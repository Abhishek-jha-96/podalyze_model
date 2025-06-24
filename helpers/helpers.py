import pandas as pd
import numpy as np

def make_ftre( X: pd.DataFrame) :
        "This function makes secondary features from the provided data"

        df = X.copy()

        df["Pub_DateTime"]  = df['Publication_Day'].astype("string") + "-" + df['Publication_Time'].astype("string")
        df["Number_of_Ads"] = df["Number_of_Ads"].fillna(0).clip(0,3).astype(np.uint8)
        df["GuestPop_Int"]  = df["Guest_Popularity_percentage"].fillna(-1).astype(np.int16)
        df["GuestPop_Dec"]  = (df["Guest_Popularity_percentage"] - df["GuestPop_Int"]).fillna(-1)
        df["Total_Pop"]     = df["Guest_Popularity_percentage"] + df["Host_Popularity_percentage"]
        df["Diff_Pop"]      = df["Guest_Popularity_percentage"] - df["Host_Popularity_percentage"]
        df["TotalPop_vs_Ads"] = np.log1p(df["Total_Pop"] ) - np.log1p(df["Number_of_Ads"])

        return df