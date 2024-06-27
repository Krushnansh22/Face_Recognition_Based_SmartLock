import DataCollection as dc
import Predict as pr

data_directory = input("Enter Directory name to store training data - ")
model_path = input("Enter model path - ")

dc.collect_and_train(data_directory,model_path)
pr.make_predictions(model_path)