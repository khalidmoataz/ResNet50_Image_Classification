from ResNet50_KNN_Preprocess_Data import preprocess_data,apply_KNN

# Preprocessing data and extract feartures

X_train,Y_train,X_validation,Y_validation = preprocess_data('101_ObjectCategories')

# Apply KNN To categorize to the only 101 categories

apply_KNN(5,X_train.Y_train)





