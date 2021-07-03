# importing libraries
import streamlit as st
import pickle
import numpy as np

# loading pickle models
lin_model = pickle.load(open('lin_model.pkl', 'rb'))
log_model = pickle.load(open('log_model.pkl', 'rb'))
svc_model = pickle.load(open('svc_model.pkl', 'rb'))


# defining a function to classify the output
def classify(num):
    if num < 0.5:
        return 'Iris-Setosa'
    elif num < 1.5:
        return 'Iris-Versicolor'
    else:
        return 'Iris-Virginica'


# main function
def main():

    # creating title header
    html_title = """
    <h1 style="text-align: center; margin-bottom: 20px;">My First Streamlit Application</h1>
    """
    st.markdown(html_title, unsafe_allow_html=True)

    # creating sub-header
    html_temp = """
    <div style="background-color: #ff5252; padding: 10px;">
    <h2 style="color: white; text-align: center;">Iris Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # creating side-bar
    models = ['Linear Regression', 'Logistic Regression', 'Support Vector Classifier']
    option = st.sidebar.selectbox('Model', models)

    # creating slider
    st.subheader(option)
    sepal_len_slider = st.slider('Select Sepal Length', 0.0, 10.0)
    sepal_width_slider = st.slider('Select Sepal Width', 0.0, 10.0)
    petal_len_slider = st.slider('Select Petal Length', 0.0, 10.0)
    petal_width_slider = st.slider('Select Petal Width', 0.0, 10.0)

    # creating classify button and displaying output
    inputs = [[sepal_len_slider, sepal_width_slider, petal_len_slider, petal_width_slider]]
    if st.button('Classify'):
        if option == 'Linear Regression':
            st.success(classify((lin_model.predict(inputs))))

        elif option == 'Logistic Regression':
            st.success(classify((log_model.predict(inputs))))

        else:
            st.success(classify(svc_model.predict(inputs)))


if __name__ == '__main__':
    main()
