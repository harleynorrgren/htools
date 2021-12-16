import json
import os
import pandas as pd


class HDataLoader:

    def __init__(self):
        pass

    def load_a_json(self, location: os.path) -> dict:
        """reads a .json file from disk"""
        if self.check_for_a_file(location):
            with open(location) as file:
                data = json.load(file)
            return data
        else:
            raise ValueError(f"file at {location} not found, try another path")

    @staticmethod
    def write_a_json(dictionary: dict, file_path: os.path):
        """writes out a dictionary to a .json object
        make sure your location is pathlike"""

        with open(file_path, "w") as file:
            json.dump(dictionary, file, indent=1)

    @staticmethod
    def check_for_a_file(file_path_str: str) -> bool:
        """returns true or false if a file exists at the specified path"""
        return os.path.exists(os.path.abspath(file_path_str))

    def load_a_text_file(self, location: os.path, sep: str = ",") -> pd.DataFrame:
        """reads a csv file"""

        if self.check_for_a_file(location):
            df = pd.read_csv(location, sep=sep)
            return df
        else:
            raise ValueError(f"file at {location} not found, try another path")
