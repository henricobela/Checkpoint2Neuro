from typing import Any
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from api.core.memoization import memo, memoize
import threading
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



class TksRequest():
    def __init__(self) -> None:
        pass


    def organize_data(self, df:pd.DataFrame) -> pd.DataFrame:
        df.drop("entry_id", axis = 1, inplace = True)
        df["created_at"] = df["created_at"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d"))
        df["date"] = np.nan * len(df)
        df["time"] = np.nan * len(df)
        df["local"] = np.nan * len(df)
        for idx, value in enumerate(df.created_at):
            df["date"][idx] = df["created_at"][idx].split("-")[0]
        for idx, value in enumerate(df.created_at):
            df["time"][idx] = df["created_at"][idx].split("-")[1]
        for idx, value in enumerate(df.created_at):
            df["local"][idx] = df["created_at"][idx].split("-")[-1]
        df.drop("created_at", axis = 1, inplace = True)
        df = df.reindex(["date", "time", "local", "field1", "field2", "field3", "field4", "field5", "field6", "field7", "field8"], axis = 1)
        df.local = df.local.apply(lambda x: 0 if x == "UTC" else 1)
        for col in df.columns:
            df[col] = df[col].astype(float)
        return df


    @memoize
    def get_data_response(self) -> pd.DataFrame:
        endpoints, all_df = [
            "https://api.thingspeak.com/channels/2167188/feeds.json", 
            "https://thingspeak.com/channels/12397/feeds.json",
            "https://thingspeak.com/channels/1785844/feeds.json",
            "https://thingspeak.com/channels/2257912/feeds.json",
            ], []
       
        for endpoint in endpoints:
            config_data = requests.get(endpoint).json()["feeds"]
            df = pd.DataFrame(config_data)
            df = self.organize_data(df)
            all_df.append(df)
        
        df = pd.concat(all_df)
        df.reset_index(drop=True, inplace=True)

        return df


class Prepare_data():
    def __init__(self) -> None:
        pass


    def prepare_data_to_model(self):
        get_df = TksRequest()
        df = get_df.get_data_response()
        df.rename(columns = {"field2": "target"}, inplace = True)
        df.fillna(df.mean(), inplace = True)
        # df.fillna(df[["field8"]].mean(), inplace = True)
        return df


    def split_data(self):
        df = self.prepare_data_to_model()
        X = df.drop(columns = ["target"])
        y = df["target"]
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)
        return train_x, test_x, train_y, test_y

    

class Train_model():
    def __init__(self) -> None:
        prep_data = Prepare_data()
        self.train_x, self.test_x, self.train_y, self.test_y = prep_data.split_data()


    def linear_model(self):
        model = LinearRegression()
        num_threads = 4
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=model.fit(self.train_x, self.train_y))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        return model


    def model_predict(self):
        y_pred = self.linear_model.predict(self.test_x)
        return y_pred



class Predict_model():
    def __init__(self) -> None:
        model = Train_model()
        self.model = model.linear_model()
    

    def real_predict(self, data_to_predict):
        # data_to_predict = TksRequest().organize_data(data_to_predict)
        return self.model.predict(data_to_predict)







