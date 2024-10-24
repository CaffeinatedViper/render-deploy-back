from uuid import UUID, uuid4
import pandas as pd
from werkzeug.datastructures.file_storage import FileStorage
import pickle
import os
import glob


class DataStore:
    def __init__(self):
        self.storePath = os.path.join(os.getcwd(), "downloads")
        os.makedirs(self.storePath, exist_ok=True)
        

        # files = glob.glob(os.path.join(self.storePath, "*"))
        # for f in files:
        #     os.remove(f)

    def store_df(self, df: pd.DataFrame) -> UUID:
        id = str(uuid4())
        file_path = os.path.join(self.storePath, id)
        with open(file_path, 'wb') as file:
            pickle.dump(df, file)
        return id

    def get_dataset(self, id: UUID) -> pd.DataFrame:
        file_path = os.path.join(self.storePath, str(id))
        try:
            with open(file_path, 'rb') as file:
                df = pickle.load(file)
            return df
        except FileNotFoundError:
            raise KeyError(f"No file found for ID: {id}")

    def store_file(self, file: FileStorage) -> UUID:
        df = pd.read_csv(file)
        print(f"DataFrame shape: {df.shape}")
        return self.store_df(df)

    def visualize(self, id: UUID) -> tuple:
        df = self.get_dataset(id)
        rowAmt = df.shape[0]
        cols = df.columns.tolist()
        head = df.head(5).values.tolist()
        tail = df.tail(5).values.tolist()
     
        head = [[str(x) if pd.notna(x) else "NaN" for x in row] for row in head]
        tail = [[str(x) if pd.notna(x) else "NaN" for x in row] for row in tail]
        return rowAmt, cols, head, tail