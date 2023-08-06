import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/emon1/OneDrive/Desktop/Machine Learning Projects/20 Deploy Machine Learning Model using Streamlit/trained_model.sav','rb'))

input_data = (0,137,40,35,168,43.1,2.288,33)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The preson is diabetic')