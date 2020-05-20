#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import time

import pandas as pd
import numpy as np

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV


# In[2]:


def Create_NN_Model(No_Features=100, No_Hidden_Layers=1, No_Hidden_Neurons=30, 
                    Hidden_Activation ="relu", No_OP_Neurons=1, 
                    Output_Activation="sigmoid", Kernel_Initializer="ones",
                    Optimizer="rmsprop", Loss='binary_crossentropy', Metrics =['accuracy']):
  
    classifier = Sequential()

  ## Input Layer
    classifier.add(Dense(No_Hidden_Neurons, activation=Hidden_Activation, 
                       kernel_initializer=Kernel_Initializer, input_dim=No_Features))
  
  ## Hidden layers
    for i in range(No_Hidden_Layers):
        classifier.add(Dense(No_Hidden_Neurons, activation=Hidden_Activation, 
                         kernel_initializer=Kernel_Initializer))
    
  ## Output Layer
    classifier.add(Dense(No_OP_Neurons, activation=Output_Activation, 
                       kernel_initializer=Kernel_Initializer))
  
    classifier.compile(optimizer =Optimizer, loss=Loss, metrics = Metrics)

    return classifier


# In[3]:


def Train_NN(NN_classifier, train_data, feature_list=[], Label_Col="Label", Batch_Size=64, Epochs=10):

    train_data.dropna()
    train_data = pd.DataFrame(np.nan_to_num(np.array(train_data)), columns = train_data.columns)
    train_data[Label_Col] = pd.to_numeric(train_data[Label_Col], errors='coerce')
    train_data = train_data.dropna(subset=[Label_Col])
  
    train_features = train_data[feature_list]    
    train_labels = train_data[Label_Col]
    train_labels = train_labels.astype('int')

    NN_classifier.fit(train_features,train_labels, batch_size=Batch_Size, epochs=Epochs)

    eval_model=NN_classifier.evaluate(train_features, train_labels)
    print("Loss: ", eval_model[0])
    print("Accuracy of the model: ", eval_model[1])

    return NN_classifier


# In[4]:


## Store trained model in a file to reuse in other codes without training again on same data

def Store_Trained_NN(NN_obj, Filepath):
  
    with open(Filepath, "wb") as file:
        pickle.dump(NN_obj, file)


# In[5]:


## Load stored trained model and returns random forest model object

def Load_Trained_NN(Filepath):
  
    with open(Filepath, "rb") as file:
        NN_obj = pickle.load(file)

    return NN_obj


# In[6]:


def Evaluate_NN(test_data, NN_Model_FilePath, feature_list=[], Label_Col="label", threshold=0.5):
  
    test_data.dropna()
    test_data = pd.DataFrame(np.nan_to_num(np.array(test_data)),  columns = test_data.columns)
    test_data[Label_Col] = pd.to_numeric(test_data[Label_Col], errors='coerce')
    test_data = test_data.dropna(subset=[Label_Col])

    test_features = test_data[feature_list]
    test_labels = test_data[Label_Col]
    test_labels = test_labels.astype('int')

    NN_obj = Load_Trained_NN(NN_Model_FilePath) 
    predictions = NN_obj.predict(test_features)
    predictions_list = [int(p[0]) for p in predictions]
      
    true_subjective = 0
    true_objective = 0
    false_subjective = 0
    false_objective = 0
    
    for i in range(len(predictions_list)):
        if predictions_list[i] >= threshold:
            predictions_list[i] = 1
            if test_labels[i] == 1:
                true_subjective += 1
            else:
                false_subjective += 1
        else:
            predictions_list[i] = 0
            if test_labels[i] == 0:
                true_objective += 1
            else:
                false_objective += 1
      
    errors = abs(predictions_list - test_labels)

  # Calculate mean absolute error (MAE)
    MAE = round(np.mean(errors), 2)
  
  ## Confusion Matrix and Classification Report
    Confusion_Matrix = confusion_matrix(test_labels,predictions_list)
    Report = classification_report(test_labels,predictions_list)
    
    print("True Subjective : ", true_subjective)
    print("True Objective : ", true_objective)
    print("False Subjective : ", false_subjective)
    print("False Objective : ", false_objective)
    
    print("Accuracy: ", (true_subjective+true_objective)/(true_subjective+true_objective+false_subjective+false_objective)*100)
  
    return MAE, Confusion_Matrix, Report


# # In[8]:


# Column_List = ["embedding"]
# Label_Col = "label"
# Vector_Size = 100
# Embedding_Cols = ["emb"+str(i) for i in range(Vector_Size)]
# # Data_Columns = Embedding_Cols
# # Data_Columns = Data_Columns.append(Label_Col)
# Column_List.append(Label_Col)

# Folder_Path = "/Users/pranjali/Downloads/Wiki_BiasDetection"

# Train_Embedding_FilePath = Folder_Path + "/Data/Task1_FinalData/Embeddings/train_emb_full.csv"
# Test_Embedding_FilePath = Folder_Path + "/Data/Task1_FinalData/Embeddings/test_emb_full.csv"
# NN_Model_FilePath =  Folder_Path + "/Saved_Models/NN/NN_Task1_Trained_Model.pkl"

# train_data_raw = pd.read_csv(Train_Embedding_FilePath, usecols=Column_List)
# test_data_raw = pd.read_csv(Test_Embedding_FilePath, usecols=Column_List)


# # In[9]:


# def Get_Embeddings(data):
    
#     Col_List = ["emb"+str(i) for i in range(100)]
#     Col_List.append("label")
    
#     Embeddings = []
    
#     for i in range(data.shape[0]):
#         row = data.iloc[i]
#         embedding_str = row["embedding"][1:-1]
#         embedding_list = embedding_str.split(',')
#         embedding = [float(s) for s in embedding_list]
#         embedding.append(int(row["label"]))
#         Embeddings.append(embedding)
        
#     return pd.DataFrame(Embeddings, columns=Col_List)

# train_data = Get_Embeddings(train_data_raw)
# test_data = Get_Embeddings(test_data_raw)


# # In[10]:


# print("train_data shape: ", train_data.shape)
# print("test_data shape: ", test_data.shape)


# # In[11]:


# train_data = train_data.sample(frac=1)


# # In[12]:


# ## Training Phase
# start_time = time.time()

# NN_Classifier = Create_NN_Model()
# NN_obj = Train_NN(NN_Classifier, train_data, Embedding_Cols, Label_Col)
# Store_Trained_NN(NN_obj, NN_Model_FilePath)

# end_time = time.time()
# print("Time required for training: ", end_time - start_time )


# # In[13]:


# start_time = time.time()

# MAE, Confusion_Matrix, Report = Evaluate_NN(test_data, NN_Model_FilePath, Embedding_Cols, Label_Col, 0.5)

# print("MEAN ABSOLUTE ERROR: ", MAE)

# print("\n")
# print("============ CONFUSION MATRIX ===============")
# print(Confusion_Matrix)

# print("\n")
# print("============ CLASSIFICATION REPORT ==============")
# print(Report)

# tn, fp, fn, tp = Confusion_Matrix.ravel()
# Accuracy = (tn+tp)/(tn + fp + fn + tp)

# print("Accuracy: ", Accuracy*100)

# end_time = time.time()
# print("Time required for testing: ", end_time - start_time )


# # In[14]:


# ## Cross Validation

# train_data.dropna()
# train_data = pd.DataFrame(np.nan_to_num(np.array(train_data)), columns = train_data.columns)
# train_data[Label_Col] = pd.to_numeric(train_data[Label_Col], errors='coerce')
# train_data = train_data.dropna(subset=[Label_Col])

# train_features = train_data[Embedding_Cols]    
# train_labels = train_data[Label_Col]
# train_labels = train_labels.astype('int')

# # create the sklearn model for the network
# model_CV = KerasClassifier(build_fn=Create_NN_Model, verbose=1)

# # we choose the initializers that came at the top in our previous cross-validation!!
# # kernel_initializer = ['random_uniform']
# # No_Hidden_Layers = [1, 2]
# # No_Hidden_Neurons= [10, 30, 50]
# # optimizer = ['adam', 'rmsprop']

# # batches = [64*x for x in range(1, 3)]
# # epochs = [50, 100, 150]

# kernel_initializer = ['random_uniform']
# No_Hidden_Layers = [1, 2]
# No_Hidden_Neurons= [10, 30, 50]
# optimizer = ['adam', 'rmsprop']

# batches = [64*x for x in range(1, 3)]
# epochs = [20, 50]

# ## We can also try different learning rates for optimizers. 
# ## For this create different objects of optimizers with different learning rates and pass list of objects

# # grid search for initializer, batch size and number of epochs
# param_grid = dict(epochs=epochs, batch_size=batches, Kernel_Initializer=kernel_initializer, 
#                  No_Hidden_Layers=No_Hidden_Layers, Optimizer=optimizer)
# grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, cv=3, n_jobs=4, refit=True, verbose=2)
# grid_result = grid.fit(train_features, train_labels)


# # In[15]:


# # print results of cross validation

# print(f'Best Accuracy for {grid_result.best_score_} using {grid_result.best_params_}')
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print(f' mean={mean:.4}, std={stdev:.4} using {param}')


# # In[16]:


# # ## Train again using best parameter values identified by cross validation and store the trained model

# # ## Training Phase
# # NN_Classifier = Create_NN_Model(No_Hidden_Layers=grid_result.best_params_.No_Hidden_Layers, No_Hidden_Neurons=grid_result.best_params_.No_Hidden_Neurons, Kernel_Initializer=grid_result.best_params_.Kernel_Initializer, Optimizer=grid_result.best_params_.Optimizer)
# # NN_obj = Train_NN(NN_Classifier, train_data, Embedding_Cols, Label_Col, Batch_Size=grid_result.best_params_.batch_size, Epochs=grid_result.best_params_.epochs)
# # Store_Trained_NN(NN_obj, NN_Model_FilePath)

# ## Store models trained with best parameters
# Store_Trained_NN(grid, NN_Model_FilePath)

# ## Evaluation of above model on validation data
# MAE, Confusion_Matrix, Report = Evaluate_NN(test_data, NN_Model_FilePath, Embedding_Cols, Label_Col, 0.5)

# print("MEAN ABSOLUTE ERROR: ", MAE)

# print("\n")
# print("============ CONFUSION MATRIX ===============")
# print(Confusion_Matrix)

# print("\n")
# print("============ CLASSIFICATION REPORT ==============")
# print(Report)

# tn, fp, fn, tp = Confusion_Matrix.ravel()
# Accuracy = (tn+tp)/(tn + fp + fn + tp)

# print("Accuracy: ", Accuracy*100)

