import sklearn.pipeline
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn import pipeline
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import math
import numpy as np


class HLazyPredict:

    def __init__(self, df: pd.DataFrame, y_var: str, random_state: int = 69, test_size: float = 0.2):
        self.df = df                                    # input df
        self.y_var = y_var                              # input y var colname
        self.random_state = random_state
        self.test_size = test_size                      # input proportion of test:train
        self.x, self.y = self.__split_df_into_x_y()     # dfs of predictors and response variables, respectively
        self.x_train, self.x_test, self.y_train, self.y_test = self.__get_test_train_split(self.x, self.y)  # x and y subdivided into train and test
        self.has_modelled = False                       # boolean flag for catching implementation errors
        self.__vlp = None                               # output of very lazy modelling run
        self.__provided_models = dict()                 # dictionary of all pipelines, keyed from model name
        self.__models = pd.DataFrame()                  # df of model performance
        self.__predictions = pd.DataFrame()             # predictions of for each row, by model attempted by vlp
        self.__pipeline = None                          # winning pipeline object selected from lazy predict
        self.model_type = None                          # string containing which model was selected
        self.__marginal_df = pd.DataFrame()             # used as a nasty hack for calculating marginal probabilities

    def __split_df_into_x_y(self) -> (pd.DataFrame, pd.Series):
        """takes a df and breaks it out into two parts, x and y"""

        if self.y_var in self.df:
            y = self.df[self.y_var]
            x = self.df.drop(self.y_var, 1)
            return x, y
        else:
            raise ValueError(f"{self.y_var} not found in df for lazy prediction...")

    def __get_test_train_split(self, x: pd.DataFrame, y: pd.Series) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
        """splits out x and y parts into train and test"""

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)
        return x_train, x_test, y_train, y_test

    def very_lazy_classifier(self) -> (pd.DataFrame, pd.DataFrame):
        """runs the lazy classifier, also stores model files into self.provided_models"""

        self.__vlp = LazyClassifier(predictions=True)
        self.__models, self.__predictions = self.__vlp.fit(self.x_train, self.x_test, self.y_train, self.y_test)
        self.has_modelled = True
        self.__get_provided_models()
        return self.__models, self.__predictions

    def very_lazy_regressor(self) -> (pd.DataFrame, pd.DataFrame):
        """runs the lazy regressor, also stores model files into self.provided_models"""

        self.__vlp = LazyRegressor(predictions=True)
        self.__models, self.__predictions = self.__vlp.fit(self.x_train, self.x_test, self.y_train, self.y_test)
        self.has_modelled = True
        self.__get_provided_models()
        return self.__models, self.__predictions

    def get_models(self) -> pd.DataFrame:
        """returns models after modelling has been done"""

        if self.has_modelled:
            return self.__models
        else:
            raise ValueError("modelling hasn't been done yet! run a model first")

    def get_predictions(self) -> pd.DataFrame:
        """returns predictions after modelling has been done"""

        if self.has_modelled:
            return self.__predictions
        else:
            raise ValueError("modelling hasn't been done yet! run a model first")

    def __get_provided_models(self) -> dict:
        """once models have been run, this pulls a dictionary of pipelines, keyed off the model names"""

        if self.has_modelled:
            self.__provided_models = self.__vlp.provide_models(self.x_train, self.x_test, self.y_train, self.y_test)
            return self.__provided_models

    def get_pipelie_object(self, which_model: str) -> pipeline:
        """returns the pipeline object you just selcted, also saves it in self.__pipeline"""

        if self.has_modelled and which_model in self.__provided_models:
            self.__pipeline = self.__provided_models[which_model]
            self.model_type = which_model
            return self.__pipeline

    def get_pipeline_coeffs(self) -> pd.DataFrame:
        """where appropriate, pull coeffs from a pipeline"""

        if self.__pipeline:

            # pull coeff information from pipeline
            coeffs = self.__pipeline.named_steps['classifier'].coef_.tolist()
            coeffs = pd.DataFrame(coeffs).transpose()
            coeffs = coeffs.rename({0: "coeff"}, axis=1)

            # take feature names from df
            feature_names = pd.DataFrame(self.x.columns)
            feature_names = feature_names.rename({0: "name"}, axis=1)

            # concat together
            coeffs_df = pd.concat([feature_names, coeffs], axis=1)
            coeffs_df = coeffs_df.sort_values(by='coeff', axis=0, ascending=True)
            print(coeffs_df.head())

            return coeffs_df

    def get_marginal_probabilities(self):
        """extensible function for getting probabilities from various pipelines
            LogisticRegression is provided here for POC"""

        self.__generate_marginal_df()
        coeffs = self.get_pipeline_coeffs()
        coeffs['p'] = 0.0000

        # todo: break this section out into a separate function
        # gets correct probability prediction function based on model type
        if self.model_type == "LinearSVC":
            self.__marginal_df = self.__get_linearsvc_probabilities()

        elif self.model_type == "LogisticRegression":
            self.__marginal_df = self.__get_logistic_probabilities(self.__marginal_df)

        for c in range(0, self.x.columns.size):
            big_prob = self.__marginal_df.at[(c*2)+1, "prob"]
            little_prob = self.__marginal_df.at[c*2, "prob"]
            print(f"{self.x.columns[c]}, {big_prob} - {little_prob} = {big_prob - little_prob}")

            coeffs.at[c, "p"] = big_prob - little_prob

        coeffs = coeffs.sort_values(["p"], ascending=True)

        return coeffs

    def __get_linearsvc_probabilities(self):
        """gets probabilities from linearSVC models using platt scaling"""
        # todo rework this with more time
        # if self.model_type == "LinearSVC":
        #     clf = CalibratedClassifierCV(self.__pipeline)
        #     clf.fit(self.x_train, self.y_train)
        #     y_proba = clf.predict_proba(self.x_test)
        #     print(y_proba)
        #     return y_proba
        # else:
        #     raise TypeError("model type is not LinearSVC")
        pass

    def __get_logistic_probabilities(self, df: pd.DataFrame):
        """returns probabilities for marginal df"""
        if self.model_type == "LogisticRegression":

            df['prob'] = self.__pipeline.predict_proba(df)[:, 1]

            return df

    def __generate_marginal_df(self):
        """makes a dataframe of 1 and 2 observations for each column"""
        number_of_columns = self.x.columns.size
        # make a df full of zeroes
        marginal_df = pd.DataFrame(np.zeros((number_of_columns*2, number_of_columns)))

        # start on row 0
        r = 0

        # nasty nested for loop
        for c in range(0, number_of_columns):
            for v in (30, 31):
                marginal_df.iloc[r, c] = v
                r += 1
            marginal_df = marginal_df.rename({c: self.x.columns[c]}, axis=1)

        # store your dirty deed
        self.__marginal_df = marginal_df

        return marginal_df



