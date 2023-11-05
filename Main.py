from RandomForest import RandomForest

rf_model = RandomForest(num_trees=10)

file_path = "iris.csv"
col_names = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'type']
rf_model.train_model(file_path, col_names)

sample_data_point = [4.1, 3.5, 1.2, 0.1] 
predicted_class = rf_model.random_forest_predict(sample_data_point)
print("Predicted Class:", predicted_class)
