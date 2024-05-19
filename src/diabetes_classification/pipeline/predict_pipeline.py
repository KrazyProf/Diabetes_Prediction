import sys
import pandas as pd
from diabetes_classification.exception.exception import CustomException
from diabetes_classification.utils.utlis import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds 
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
                    PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
                    GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age):
            self.HighBP = HighBP
            self.HighChol = HighChol
            self.CholCheck = CholCheck
            self.BMI = BMI
            self.Smoker = Smoker
            self.Stroke = Stroke
            self.HeartDiseaseorAttack = HeartDiseaseorAttack
            self.PhysActivity = PhysActivity
            self.Fruits = Fruits
            self.Veggies = Veggies
            self.HvyAlcoholConsump = HvyAlcoholConsump
            self.AnyHealthcare = AnyHealthcare
            self.NoDocbcCost = NoDocbcCost
            self.GenHlth = GenHlth
            self.MentHlth = MentHlth
            self.PhysHlth = PhysHlth
            self.DiffWalk = DiffWalk
            self.Sex = Sex
            self.Age = Age

    def get_data_as_data_frame(self):
        try:
            data_dict = {
                "HighBP": [self.HighBP],
                "HighChol": [self.HighChol],
                "CholCheck": [self.CholCheck],
                "BMI": [self.BMI],
                "Smoker": [self.Smoker],
                "Stroke": [self.Stroke],
                "HeartDiseaseorAttack": [self.HeartDiseaseorAttack],
                "PhysActivity": [self.PhysActivity],
                "Fruits": [self.Fruits],
                "Veggies": [self.Veggies],
                "HvyAlcoholConsump": [self.HvyAlcoholConsump],
                "AnyHealthcare": [self.AnyHealthcare],
                "NoDocbcCost": [self.NoDocbcCost],
                "GenHlth": [self.GenHlth],
                "MentHlth": [self.MentHlth],
                "PhysHlth": [self.PhysHlth],
                "DiffWalk": [self.DiffWalk],
                "Sex": [self.Sex],
                "Age": [self.Age]
            }
            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)
