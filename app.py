import numpy as np
import pickle
import streamlit

# loaded the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))

# creating a functoin for prediction
def diabates_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The preson is diabetic'

def main():
    # giving a title
    streamlit.title('Diabates Prediction Web App')
    
    # getting the input data from the user  
    Pregnancies = streamlit.text_input('Numer of Pregnencies: ')
    Glucose = streamlit.text_input('Glucose lavel: ')
    BloodPressure = streamlit.text_input('Blood Pressure value: ')
    SkinThickness = streamlit.text_input('Skin Thickness value: ')
    Insulin = streamlit.text_input('Insulin lavel: ')
    BMI = streamlit.text_input('BMI value: ')
    DiabetesPedigreeFunction = streamlit.text_input('Diabetes Pedigree Function value: ')
    Age = streamlit.text_input('Age of the person: ')
    
    # code for prediction
    diagnosis = ''
    
    # create a button for prediction
    if streamlit.button('Diabetes Test Result'):
        diagnosis = diabates_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    streamlit.success(diagnosis)
    
if __name__ == '__main__':
    main()
