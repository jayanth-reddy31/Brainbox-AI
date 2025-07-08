# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 23:08:50 2025

@author: USER
"""
import pickle
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from streamlit_option_menu import option_menu





#loading the saved models
gender_classification = load_model('gender_model_custom_model.h5',custom_objects={"YourCustomLayer": YourCustomLayer})

big_mart_sales = pickle.load(open('big_mart_sales.sav','rb'))

calories_burnt = pickle.load(open('calories_burnt_model.sav', 'rb'))

car_price_prediction = pickle.load(open('car_price_pred_model.sav', 'rb'))

credit_card_fraud = pickle.load(open('credict_card_fraud_model.sav','rb'))

gold_price_model = pickle.load(open('Gold_price_prediction_model.sav','rb'))

house_price_model = pickle.load(open('House_price_prediction_model.sav','rb'))

loan_prediction = pickle.load(open('Loan_prediction_model.sav','rb'))

medical_insurance = pickle.load(open('Medical_cost_prediction_model.sav','rb'))

#for movie recommandation the file is downloaded in .pkl extension since it return any datatype, but .sav return function always,
#which is not iterable, since for movie recommandation system we uses cosine similarity which returns a numpy array, to find
#similar movies we need to iterate the matrix so it is important to file return numpy array
movie_recommandation = pickle.load(open('movie_recommendation_similarity_model.pkl','rb'))

rock_vs_mine = pickle.load(open('rockvsmine_prediction_model.sav','rb'))

spam_mail = pickle.load(open('spam_mail_detection.sav','rb'))

titanic_survival = pickle.load(open('Titanic_survival_prediction_model.sav','rb'))

wine_quality = pickle.load(open('wine_quality_prediction_model.sav','rb'))

diabetes_model = pickle.load(open('diabetes_trained_model.sav','rb'))

heart_disease_model =  pickle.load(open('heart_training_model.sav', 'rb'))

parkinsons_model = pickle.load(open('parkinsons_training_model.sav', 'rb'))

breast_cancer_model = pickle.load(open('breast_cancer_training_model.sav', 'rb'))


#sidebar for navigation

with st.sidebar:
    selected = option_menu("All in one Machine Learning Projects",
                           
                           ['Gender classification',
                            'Big Mart Sales Prediction',
                            'Calories Burnt Prediction',
                            'Car Price Prediction',
                            'Credit Card Prediction',
                            'Gold Price Prediction',
                            'House Price Prediction',
                            'Loan Prediction',
                            'Medical Insurance Cost',
                            'Movie Recommendation System',
                            'Rock vs mine Prediction',
                            'Spam mail Detection',
                            'Titanic Survival Prediction',
                            'Wine Quality Prediction',
                            'Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Breast Cancer Prediction'],
                           
                           icons = ['people','amazon','person-walking','car-front','credit-card','currency-rupee','house','bank','hospital',
                                    'film','minecart-loaded','mailbox2','water','cup','activity','heart','person','person-standing-dress'],  #these are icons from bootstrap, streamlit supports bootstrap
                           
                           default_index = 0) # when web page is run the default page is diabetes prediction
    
st.subheader("Made with ❤️ by Jayanth")



#Gender Classification
if(selected == 'Gender classification'):
    #page title
    st.title('📊 Gender Classification using CNN')
    st.text("This app classifies a face image to Male or Female.")
    st.write("Upload a face image to classify as Male or Female")
    # Set target size (must match model input)
    IMG_SIZE = (224, 224)  # Change this if your model used a different input size
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        img_resized = img.resize(IMG_SIZE)
        img_array = image.img_to_array(img_resized)
        img_array = img_array / 255.0  # Normalize if your model was trained that way
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = gender_classification.predict(img_array)
        gender = "Male" if prediction[0][0] > 0.5 else "Female"

        st.write(f"**Predicted Gender:** {gender}")

    
#Big mart sales prediction
if (selected == 'Big Mart Sales Prediction'):
    #page title
    st.title('📊 Big Mart Sales Predictor using XGB')
    st.text("This app predicts sales based on various features.")
    st.markdown("**Don't have values? TRY THESE INPUTS** ")
    st.markdown("156,	9.30,	0,	0.016047,	4,	249.8092,	9,	1999,	1,	0,	1,	3735.1380") 
    st.markdown("662	17.50,	0,	0.016760,	10,	141.6180,	9,	1999,	1,	0,	1,	2097.2700")


    #getting the input data from the user
    #columns for input fields

    
    #the order must be same as in the data set
    
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Item_Identifier = st.text_input('Item Identifier Number')
        
    with col2:
        Item_Weight = st.text_input('Item weight')
        
    with col3:
        Item_Fat_Content = st.text_input('Low fat=0, regular=1')
        
    with col1:
        Item_Visibility = st.text_input('Item Visibility')
    
    with col2:
        Item_Type = st.text_input('Item type')
        
    with col3:
        Item_MRP = st.text_input('Item MRP')
        
    with col1:
        Outlet_Identifier = st.text_input('Outlet Identifier')
    
    with col2:
        Outlet_Establishment_Year = st.text_input('Outlet Establishment year')
        
    with col3:
         Outlet_Size = st.text_input('Outlet size')
         
    with col1:
         Outlet_Location_Type = st.text_input('Outlet location type')
     
    with col2:
         Outlet_Type = st.text_input('outlet type')
    
    with col3:
        Item_Outlet_Sales = st.text_input('Item outlet sales')
    
    

    
    #code for prediction
    sale_prediction = ''  #empty string to store the end result
    
    #creating a button for prediction
    if st.button('Predict sales'):
        sale = [Item_Identifier, Item_Weight,Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Identifier, Outlet_Establishment_Year,
                     Outlet_Size,Outlet_Location_Type, Outlet_Type, Item_Outlet_Sales]
        
        
        #to convert the text/string data into numeric data
        sale = [float(x) for x in sale]

        sale_pred = big_mart_sales.predict([sale])
        #the columns names are in 2 square brackets to tell the model that we are predicting for one data point

        # Display the prediction
        sale_prediction = f"Predicted Sales: ₹{sale_pred[0]:,.2f}"

    st.success(sale_prediction)

#calories burnt prediction
if (selected == 'Calories Burnt Prediction'):
    #page title
    st.title('📊 Calories Burnt Predictor using XGRegressor')
    st.text("This app predicts the number of calories burnt during exercise.")
    st.markdown("**Don't have values? TRY THESE INPUTS** ")
    st.markdown("14733363,	1,	68,	190.0,	94.0,	29.0,	105.0,	40.8   -->*output: 231*") #231 calories male
    st.markdown("14861698,	0,	20,	166.0,	60.0,	14.0,	94.0,	40.3   -->*output: 66*") #66 calories female


    #getting the input data from the user
    #columns for input fields
    #User_ID	Gender	Age	Height	Weight	Duration	Heart_Rate	Body_Temp

    
    #the order must be same as in the data set
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        User_ID = st.text_input('User ID')
        
    with col2:
        Gender = st.text_input('Gender (Male:1, Female:0)')
        
    with col1:
        Age = st.text_input('Age')
        
    with col2:
        Height = st.text_input('Height')
        
    with col1:
        Weight = st.text_input('Weight')
    
    with col2:
        Duration = st.text_input('Duration')
        
    with col1:
        Heart_Rate = st.text_input('Heart Rate')
        
    with col2:
        Body_Temp = st.text_input('Body_Temp')
    
    
    #code for prediction
    calories_prediction = ''  #empty string to store the end result
    
    #creating a button for prediction
    if st.button('Predict calories'):
        calories = [User_ID,	Gender,	Age,	Height,	Weight,	Duration,	Heart_Rate,	Body_Temp]
        #the columns names are in 2 square brackets to tell the model that we are predicting for one data point
        
        #to convert the text/string data into numeric data
        calories = [float(x) for x in calories]

        calories_pred = calories_burnt.predict([calories])

        # Display the prediction
        calories_prediction = f"Predicted Calories: {calories_pred[0]:,.2f}"

    st.success(calories_prediction)


#car price prediction
if (selected == 'Car Price Prediction'):
    #page title
    st.title('🚗 Car Price Predictor using lasso Regression')
    st.text("This app aims to build Lasso regression models to predict car prices.")
    st.markdown("**Don't have values? TRY THESE INPUTS** ")
    st.markdown("2014,	5.59,	27000,	0,	0,	0,	0   -->*output: 3.35*")
    st.markdown("2017,	9.85,	6900,	0,	0,	0,	0   -->*output: 7.25*")


    #getting the input data from the user
    #columns for input fields
    #

    #the order must be same as in the data set
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        Year = st.text_input('Year')
        
    with col2:
        Present_Price = st.text_input('Present_Price')
        
    with col1:
        Kms_Driven = st.text_input('Kms Driven')
        
    with col2:
        Fuel_Type = st.text_input('Fuel Type (petrol:0, diesel:1, CNG:2)')
        
    with col1:
        Seller_Type = st.text_input('Seller Type (Dealer:0, Individual:1)')
    
    with col2:
        Transmission = st.text_input('Transmission (Manual:0, Automatic:1)')
        
    with col1:
        Owner = st.text_input('Owner')

    
    
    #code for prediction
    car_price_pred = ''  #empty string to store the end result
    
    #creating a button for prediction
    if st.button('Predict car price'):
        car = [Year,	Present_Price,	Kms_Driven,	Fuel_Type,	Seller_Type,	Transmission,	Owner]
        #the columns names are in 2 square brackets to tell the model that we are predicting for one data point
        
        #to convert the text/string data into numeric data
        car = [float(x) for x in car]

        car_pred = car_price_prediction.predict([car])

        # Display the prediction
        car_price_pred= f"Predicted Car price: {car_pred[0]:,.2f}L"

    st.success(car_price_pred)



#credit card fraud detection page
if (selected == 'Credit Card Prediction'):
    #page title
    st.title('Fraud Credit card predictor using Logistic regression')
    st.text("This app is a classification model to detect fraudulent credit card transactions.")
    st.text("The features on which is trained on PCA(a dimensianlity reduction) features")
    st.markdown("**Don't have values? TRY THESE INPUTS** ")
    st.markdown("""1,-0.966271711572087,-0.185226008082898,1.79299333957872,-0.863291275036453,-0.0103088796030823,
                1.24720316752486,0.23760893977178,0.377435874652262,-1.38702406270197,-0.0549519224713749,-0.226487263835401,
                0.178228225877303,0.507756869957169,-0.28792374549456,-0.631418117709045,-1.0596472454325,-0.684092786345479,
                1.96577500349538,-1.2326219700892,-0.208037781160366,-0.108300452035545,0.00527359678253453,-0.190320518742841,
                -1.17557533186321,0.647376034602038,-0.221928844458407,0.0627228487293033,0.0614576285006353,123.5  -->*output: legit*""") #231 calories male
    #getting the input data from the user
    #columns for input fields
    
    #the order must be same as in the data set
    
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Time = st.text_input('Time')

    with col2:
        V1 = st.text_input('V1')

    with col3:
        V2 = st.text_input('V2')

    with col1:
        V3 = st.text_input('V3')

    with col2:
        V4 = st.text_input('V4')

    with col3:
        V5 = st.text_input('V5')

    with col1:
        V6 = st.text_input('V6')

    with col2:
        V7 = st.text_input('V7')

    with col3:
        V8 = st.text_input('V8')

    with col1:
        V9 = st.text_input('V9')

    with col2:
        V10 = st.text_input('V10')

    with col3:
        V11 = st.text_input('V11')

    with col1:
        V12 = st.text_input('V12')

    with col2:
        V13 = st.text_input('V13')

    with col3:
        V14 = st.text_input('V14')

    with col1:
        V15 = st.text_input('V15')

    with col2:
        V16 = st.text_input('V16')

    with col3:
        V17 = st.text_input('V17')

    with col1:
        V18 = st.text_input('V18')

    with col2:
        V19 = st.text_input('V19')

    with col3:
        V20 = st.text_input('V20')

    with col1:
        V21 = st.text_input('V21')

    with col2:
        V22 = st.text_input('V22')

    with col3:
        V23 = st.text_input('V23')

    with col1:
        V24 = st.text_input('V24')

    with col2:
        V25 = st.text_input('V25')

    with col3:
        V26 = st.text_input('V26')

    with col1:
        V27 = st.text_input('V27')

    with col2:
        V28 = st.text_input('V28')

    with col3:
        Amount = st.text_input('Amount')
    
    
    #code for prediction
    fraud = ''  #empty string to store the end result
    
    #creating a button for prediction
    if st.button('Test the Transaction'):
        credit_pred =  ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
            'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

        #the columns names are in 2 square brackets to tell the model that we are predicting for one data point
        
        #to convert the text/string data into numeric data
        credit_pred = [float(x) for x in credit_pred]

        credit_prediction = credit_card_fraud.predict([credit_pred])

        if(credit_prediction[0] == 1):
            fraud = 'The Transaction in fraud'
        else:
            fraud = 'The Transaction is legit'

    st.success(fraud)
    st.write("Dataset Link : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")



#Gold pricce predictiion page
if (selected == 'Gold Price Prediction'):
    #page title
    st.title('Gold Price Predictor using Random Forest Regressor')
    st.text("This regression model to predict gold prices")
    st.markdown("**Don't have values? TRY THESE INPUTS** ")
    st.markdown("2723.070068,	14.4100,	15.7400,	1.191753  --> **output** : 125.180")
    st.markdown("1447.160034,	78.470001,	15.180,	1.471692  --> **output** : 84.860001")
    
    #getting the input data from the user
    #columns for input fields
    
    #the order must be same as in the data set


    col1, col2 = st.columns(2)

    with col1:
        SPX = st.text_input('S&P 500 Index Value')

    with col2:
        USO = st.text_input('United States Oil Fund (USO) Value')
        
    with col1:
        SLV = st.text_input('iShares Silver Trust (SLV) Value')

    with col2:
        EUR_USD = st.text_input('Euro to US Dollar Exchange Rate (EUR/USD)')

    
    #code for prediction
    gold_price = ''  #empty string to store the end result
    
    #creating a button for prediction
    if st.button('Predict the Gold Price'):
        gold = [SPX, USO, SLV, EUR_USD]
        #the columns names are in 2 square brackets to tell the model that we are predicting for one data point
        
        #to convert the text/string data into numeric data
        gold = [float(x) for x in gold]

        gold_pred = gold_price_model.predict([gold])

        # Display the prediction
        gold_price = f"Predicted Gold price: {gold_pred[0]:,.2f}"

    st.success(gold_price)


#House pricce predictiion page
if (selected == 'House Price Prediction'):
    #page title
    st.title('California House Price Predictor using XGB Regressor')
    st.text("This build a regression model to predict house prices.")
    st.markdown("**Don't have values? TRY THESE INPUTS** ")
    st.markdown("8.3252,	41.0,	6.984127,	1.023810,	322.0,	2.555556,	37.88,	-122.23, --> **output** : 4.526 ")
    st.markdown("3.8462,	52.0,	6.281853,	1.081081,	565.0,	2.181467,	37.85,	-122.25,  --> **output** : 3.422")
    
    #getting the input data from the user
    #columns for input fields
    
    #the order must be same as in the data set


    col1, col2 = st.columns(2)

    with col1:
        MedInc = st.text_input('Median Income in Block (MedInc)')
        
    with col2:
        HouseAge = st.text_input('Median House Age in Block (HouseAge)')

    with col1:
        AveRooms = st.text_input('Average Number of Rooms (AveRooms)')

    with col2:
        AveBedrms = st.text_input('Average Number of Bedrooms (AveBedrms)')

    with col1:
        Population = st.text_input('Block Population (Population)')

    with col2:
        AveOccup = st.text_input('Average Occupants per Household (AveOccup)')

    with col1:
        Latitude = st.text_input('Block Latitude')

    with col2:
        Longitude = st.text_input('Block Longitude')

    
    #code for prediction
    house_price = ''  #empty string to store the end result
    
    #creating a button for prediction
    if st.button('Predict house Price'):
        house = [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]

        #the columns names are in 2 square brackets to tell the model that we are predicting for one data point
        
        #to convert the text/string data into numeric data
        house = [float(x) for x in house]

        house_pred = house_price_model.predict([house])

        # Display the prediction
        house_price = f"Predicted Gold price: {house_pred[0]:,.2f}"

    st.success(house_price)


#Loan prediction prediction page
if (selected == 'Loan Prediction'):
    #page title
    st.title('Loan Prediction using SVM ')
    st.text("This project aims to build a classification model to predict whether a loan will be approved.")
    st.markdown("**Don't have values? TRY THESE INPUTS** ")
    st.markdown("60, 1, 25.84, 0, 1, 2	 --> **output** : 11557 ")
    st.markdown("33,	0,	22.705,	0,	1,	2  --> **output** : 21984.47061 ")
    
    #getting the input data from the user
    #columns for input fields
    #the order must be same as in the data set


    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.selectbox('Gender M(1), F(0)', ['0','1'])

    with col2:
        Married = st.selectbox('Married Y(1), N(0)', ['1','0'])

    with col3:
        Dependents = st.selectbox('Number of Dependents', ['0', '1', '2', '4'])

    with col1:
        Education = st.selectbox('Education Graduate(1), Non Graduate(0)', ['0','1'])

    with col2:
        Self_Employed = st.selectbox('Self Employed Y(1), N(0)', ['0','1'])

    with col3:
        ApplicantIncome = st.text_input('Applicant Income')

    with col1:
        CoapplicantIncome = st.text_input('Coapplicant Income')

    with col2:
        LoanAmount = st.text_input('Loan Amount')

    with col3:
        Loan_Amount_Term = st.text_input('Loan Amount Term (in days)')

    with col1:
        Credit_History = st.selectbox('Credit History', ['1', '0'])

    with col2:
        Property_Area = st.selectbox('Property Area Urban(2), Semiurban(1), Rural(0)', ['0','1','2'])





        
    #code for prediction
    loan_price = ''  #empty string to store the end result
    
    #creating a button for prediction
    if st.button('Predict loan'):
        loan = [Gender,	Married,	Dependents,	Education,	Self_Employed,	ApplicantIncome,	CoapplicantIncome,	LoanAmount,	Loan_Amount_Term,	Credit_History,	Property_Area	]

        #the columns names are in 2 square brackets to tell the model that we are predicting for one data point
        
        #to convert the text/string data into numeric data
        loan = [float(x) for x in loan]

        loan_pred = loan_prediction.predict([loan])

        loan_price = f"Predicted loan status: {loan_pred[0]:,.2f}"

    st.success(loan_price)



#medical insurance price prediction prediction page
if (selected == 'Medical Insurance Cost'):
    #page title
    st.title('Medical Insurance Cost Prediction using Linear Regression')
    st.text("This project aims to build a regression model to predict medical insurance costs.")
    st.markdown("**Don't have values? TRY THESE INPUTS** ")
    st.markdown("60, 1, 25.84, 0, 1, 2	 --> **output** : 11557 ")
    st.markdown("33,	0,	22.705,	0,	1,	2  --> **output** : 21984.47061 ")
    
    #getting the input data from the user
    #columns for input fields
    #the order must be same as in the data set


    col1, col2 = st.columns(2)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.selectbox('Sex M(0), F(1)', ['0', '1'])

    with col1:
        bmi = st.text_input('BMI')

    with col2:
        children = st.text_input('Number of Children')

    with col1:
        smoker = st.selectbox('Smoker Y(1), N(0)', ['0', '1'])

    with col2:
        region = st.selectbox('Region Northeast(3), Northwest(2), Southeast(1), Southwest(0)', ['0','1','2','3'])



        
    #code for prediction
    insurance_price = ''  #empty string to store the end result
    
    #creating a button for prediction
    if st.button('Predict Insurance'):
        med = [age,	sex,	bmi,	children,	smoker,	region]

        #the columns names are in 2 square brackets to tell the model that we are predicting for one data point
        
        #to convert the text/string data into numeric data
        med = [float(x) for x in med]

        med_pred = medical_insurance.predict([med])

        insurance_price = f"Predicted Insurance price: {med_pred[0]:,.2f}"

    st.success(insurance_price)




#movie recommendation system page
import difflib

# Movie Recommendation System Page
if selected == 'Movie Recommendation System':
    # Page title
    st.title('🎬 Movie Recommendation System using Cosine Similarity')
    st.text("Get top 10 movie recommendations based on your favorite movie.")
    st.markdown("**Don't have movie name? TRY THESE movie** ")
    st.markdown("""
                - **The Polar Express**  
                - **Independence Day: Resurgence**  
                - **How to Train Your Dragon**  
                - **Terminator 3: Rise of the Machines**  
                - **Guardians of the Galaxy**  
                - **Interstellar**  
                - **Inception**
                """)

    # Load data
    movies_data = pickle.load(open('E:/ML projects/all in one/title_index.pkl', 'rb'))
    movie_name = st.text_input("Enter your favourite movie name")

    if st.button("Recommend"):
        list_of_title = movies_data['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_title)

        if find_close_match:
            close_match = find_close_match[0]

            index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

            similarity_score = list(enumerate(movie_recommandation[index_of_the_movie]))
            sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

            st.subheader(f"Top 10 Recommendations for: **{close_match}**")
            i = 1
            for movie in sorted_similar_movies[1:11]:  # Skip the first match (same movie)
                index = movie[0]
                title_from_index = movies_data[movies_data.index == index]['title'].values[0]
                st.write(f"{i}. {title_from_index}")
                i += 1
        else:
            st.warning("No close match found. Please check the movie name.")



#rock vs mine predictiion page
if (selected == 'Rock vs mine Prediction'):
    #page title
    st.title('Rock vs Mine Prediction using Logistic Regression')
    st.text("This project aims to build a Classification model to predict whether the substance is rock or mine.")
    st.markdown("**Don't have values? TRY THESE INPUTS** ")

    #getting the input data from the user
    #columns for input fields
    #the order must be same as in the data set
    
    

    # Paste your feature values
    r_values = [0.0317, 0.0956, 0.1321, 0.1408, 0.1674, 0.171, 0.0731, 0.1401, 0.2083, 0.3513, 0.1786, 0.0658, 0.0513, 0.3752, 0.5419, 0.544, 0.515, 0.4262, 0.2024, 0.4233, 0.7723, 0.9735, 0.939, 0.5559, 0.5268, 0.6826, 0.5713, 0.5429, 0.2177, 0.2149, 0.5811, 0.6323, 0.2965, 0.1873, 0.2969, 0.5163, 0.6153, 0.4283, 0.5479, 0.6133, 0.5017, 0.2377, 0.1957, 0.1749, 0.1304, 0.0597, 0.1124, 0.1047, 0.0507, 0.0159, 0.0195, 0.0201, 0.0248, 0.0131, 0.007, 0.0138, 0.0092, 0.0143, 0.0036, 0.0103]
    
    m_values = [0.0394, 0.042, 0.0446, 0.0551, 0.0597, 0.1416, 0.0956, 0.0802, 0.1618, 0.2558, 0.3078, 0.3404, 0.34, 0.3951, 0.3352, 0.2252, 0.2086, 0.2248, 0.3382, 0.4578, 0.6474, 0.6708, 0.7007, 0.7619, 0.7745, 0.6767, 0.7373, 0.7834, 0.9619, 1, 0.8086, 0.5558, 0.5409, 0.4988, 0.3108, 0.2897, 0.2244, 0.096, 0.2287, 0.3228, 0.3454, 0.3882, 0.324, 0.0926, 0.1173, 0.0566, 0.0766, 0.0969, 0.0588, 0.005, 0.0118, 0.0146, 0.004, 0.0114, 0.0032, 0.0062, 0.0101, 0.0068, 0.0053, 0.0087]

    st.title("🎛 Feature Selector from R & M Data")

    st.markdown("Select values from each column for both **R** and **M** class samples.")

    # Create columns in groups of 3 for better layout
    for i in range(0, 60, 3):
        cols = st.columns(3)
        
        for j in range(3):
            if i + j < 60:
                with cols[j]:
                    col_name = str(i + j)
                    r_val = r_values[i + j]
                    m_val = m_values[i + j]
                    
                    st.selectbox(
                        f"Col {col_name}",
                        options=[r_val, m_val],
                        format_func=lambda x: f"R: {x}" if x == r_val else f"M: {x}"
                        )


    #code for prediction
    rock_mine_result = ''  #empty string to store the end result
    
    #creating a button for prediction
    if st.button('Test the object'):
        rm_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                50, 51, 52, 53, 54, 55, 56, 57, 58, 59 ]
        #the columns names are in 2 square brackets to tell the model that we are predicting for one data point
        
        #to convert the text/string data into numeric data
        rm_data = [float(x) for x in rm_data]

        rm_prediction = rock_vs_mine.predict([rm_data])

        if(rm_prediction[0] == 1):
            rock_mine_result = 'The Object is Mine'
        else:
            rock_mine_result = 'The Object is Rock'

    st.success(rock_mine_result)
    st.write("Data set Link: https://www.kaggle.com/datasets/mayurdalvi/sonar-mine-dataset")


#Spam mail Detection page
if (selected == 'Spam mail Detection'):
    #page title
    st.title('Spam Mail detection using Logistic Regression')
    st.text("This project aims to build a Classification model to predict whether the mail is spam or ham.")

    mail_text = st.text_input("Enter a mail text")
    
    #get extracted feature .pkl file
    feature_extraction = pickle.load(open('E:/ML projects/all in one/feature_extraction.pkl','rb'))
    
    mail_result=''
    if st.button("Test the Mail"):
        
        input_data_features = feature_extraction.transform([mail_text])

        prediction = spam_mail.predict(input_data_features)


        if(prediction[0]==1):
            mail_result = "It is a Ham mail"
        else:
            mail_result = "It is a spam mail"
        
    st.success(mail_result)


#Titanic survial predictiion page
if (selected == 'Titanic Survival Prediction'):
    #page title
    st.title('Titanic Survival Prediction using Logistic Regression')
    st.text("This project aims to build a Classification model to predict whether the substance is rock or mine.")
    st.markdown("**Don't have values? TRY THESE INPUTS** ")
    st.markdown("3,	0,	22.0,	1,	0,	7.2500,	0	 --> **output** : Not survived ")
    st.markdown("1, 1,	38.0,	1,	0, 71.2833,	1 --> **output** : Survived ")
    
    #getting the input data from the user
    #columns for input fields
    #the order must be same as in the data set


    col1, col2 = st.columns(2)

    with col1:
        # Pclass: 1 = 1st class, 2 = 2nd class, 3 = 3rd class
        Pclass = st.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', ['1', '2', '3'])
        
    with col2:
        # Sex: 0 = male, 1 = female
        Sex = st.selectbox('Sex (Male = 0, Female = 1)', ['0', '1'])

    with col1:
        # Age: numeric input
        Age = st.text_input('Age')

    with col2:
        # SibSp: Number of siblings/spouses aboard
        SibSp = st.text_input('Siblings/Spouses aboard (SibSp)')

    with col1:
        # Parch: Number of parents/children aboard
        Parch = st.text_input('Parents/Children aboard (Parch)')

    with col2:
        # Fare: ticket fare
        Fare = st.text_input('Fare')

    with col1:
        # Embarked: 0 = S, 1 = C, 2 = Q
        Embarked = st.selectbox('Port of Embarkation (S = 0, C = 1, Q = 2)', ['0', '1', '2'])

        
    #code for prediction
    survival = ''  #empty string to store the end result
    
    #creating a button for prediction
    if st.button('Predict'):
        data = [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]


        #the columns names are in 2 square brackets to tell the model that we are predicting for one data point
        
        #to convert the text/string data into numeric data
        data = [float(x) for x in data]

        titanic_pred = titanic_survival.predict([data])


        if(titanic_pred[0] == 1):
            survival = 'The person is Survived'
        else:
            survival = 'The person is not Survived'

    st.success(survival)


#wine quality predictiion page
if (selected == 'Wine Quality Prediction'):
    #page title
    st.title('Wine Quality Prediction using Random Forest Classifier')
    st.text("This project aims to build a Classification model to predict whether the wine is pure or not.")
    st.markdown("**Don't have values? TRY THESE INPUTS** ")
    st.markdown("7.3,	0.65,	0,	1.2,	0.065,	15,	21,	0.9946,	3.39,	0.47,	10	 --> **output** : Good quality")
    st.markdown("6.7,	0.58, 0.08,	1.8,	0.097,	15,	65,	0.9959,	3.28,	0.54,	9.2--> **output** : Bad quality ")
    
    #getting the input data from the user
    #columns for input fields
    #the order must be same as in the data set



    col1, col2, col3 = st.columns(3)

    with col1:
        fixed_acidity = st.text_input('Fixed Acidity')

    with col2:
        volatile_acidity = st.text_input('Volatile Acidity')

    with col3:
        citric_acid = st.text_input('Citric Acid')

    with col1:
        residual_sugar = st.text_input('Residual Sugar')

    with col2:
        chlorides = st.text_input('Chlorides')

    with col3:
        free_sulfur_dioxide = st.text_input('Free Sulfur Dioxide')

    with col1:
        total_sulfur_dioxide = st.text_input('Total Sulfur Dioxide')

    with col2:
        density = st.text_input('Density')

    with col3:
        pH = st.text_input('pH')

    with col1:
        sulphates = st.text_input('Sulphates')

    with col2:
        alcohol = st.text_input('Alcohol')

            
    #code for prediction
    quality = ''  #empty string to store the end result
    
    #creating a button for prediction
    if st.button('Predict the quality'):
        wine = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]



        #the columns names are in 2 square brackets to tell the model that we are predicting for one data point
        
        #to convert the text/string data into numeric data
        wine = [float(x) for x in wine]

        quality_pred = wine_quality.predict([wine])


        if(quality_pred[0] == 1):
            quality = 'Good quality wine'
        else:
            quality = 'Bad quality wine'

    st.success(quality)




#Diabetes predictiion page
if (selected == 'Diabetes Prediction'):
    #page title
    st.title('Diabetes prediction using SVM')
    st.text("This project aims to build a Classification model to detect the presence of diabetes.")
    st.markdown("**Don't have values? TRY THESE INPUTS** ")
    st.markdown("6,	148,	72,	35,	0,	33.6,	0.627,	50	 --> **output** : present")
    st.markdown("1,	89,	66,	23,	94,	28.1,	0.167,	21--> **output** : Absent ")
    
    #getting the input data from the user
    #columns for input fields
    #the order must be same as in the data set
    
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
        
    with col3:
        BloodPresure = st.text_input('Blood Pressure value')
        
    with col1:
        SkinThicknes = st.text_input('Skin thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin value')
        
    with col3:
        BMI = st.text_input('BMI value')
        
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes predigree function value')
    
    with col2:
        Age = st.text_input('Age of the person')


    #code for prediction
    diab_diagnosis = ''  #empty string to store the end result
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diab_pred = [Pregnancies, Glucose,BloodPresure, SkinThicknes, Insulin, BMI, DiabetesPedigreeFunction, Age]
        #the columns names are in 2 square brackets to tell the model that we are predicting for one data point
        
        #to convert the text/string data into numeric data
        diab_pred = [float(x) for x in diab_pred]

        diab_prediction = diabetes_model.predict([diab_pred])

        if(diab_prediction[0] == 1):
            diab_diagnosis = 'The person is Diabetic'
        else:
            diab_diagnosis = 'The person is not Diabetic'

    st.success(diab_diagnosis)



#heart disease prediction page
if(selected == 'Heart Disease Prediction'):
    st.title('Heart Disease prediction using Logistic regression')
    st.text("This project aims to build a Classification model to detect the presence of heart disease.")
    st.markdown("**Don't have values? TRY THESE INPUTS** ")
    st.markdown("41,0,1,130,204,0,0,172,0,1.4,2,0,2	 --> **output** : present")
    st.markdown("68,	1,	0,	144,	193,	1,	1,	141,	0,	3.4,	1,	2,	3--> **output** : Absent ")
    
    #getting the input data from the user
    #columns for input fields
    #the order must be same as in the data set
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input("Age of the person")
        
    with col2:
        sex = st.text_input("Gender of the person")
    
    with col3:
        cp = st.text_input("Chest pain type")
        
    with col1:
        trestbps = st.text_input("Resting blood pressure")
        
    with col2:
        chol = st.text_input("Serum cholestrol")
        
    with col3:
        fbs = st.text_input("Fasting blood pressure")
        
    with col1:
        restecg = st.text_input("Resting electrocardiographic")
        
    with col2:
        thalach = st.text_input("Maximum heart rate")
        
    with col3:
        exang = st.text_input("Exercise induced angina")
        
    with col1:
        oldpeak = st.text_input("ST depresion induced by exercise relative to rest")
        
    with col2:
        slope = st.text_input("The slope of the peak exercise")
        
    with col3:
        ca = st.text_input("Number of major vessels")
        
    with col1:
        thal = st.text_input("Normal/Defect/Reversable")
        
        
    #code for prediction
    heart_diagnosis = ''  #empty string to store the end result
    
    #creating a button for prediction
    if st.button('Heart Disease Test Result'):
        heart_pred = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        #the columns names are in 2 square brackets to tell the model that we are predicting for one data point
        heart_pred = [float(x) for x in heart_pred]

        heart_prediction = heart_disease_model.predict([heart_pred])
        
        if(heart_prediction[0] == 1):
            heart_diagnosis = 'The person is Diabetic'
        else:
            heart_diagnosis = 'The person is not Diabetic'

    st.success(heart_diagnosis)
    
    
    
    
    
    
#parkinsons prediction page
if(selected =='Parkinsons Prediction'):
    st.title('Parkinsons prediction using SVM')
    st.text("This project aims to build a Classification model to detect the presence of parkiinsons disease.")
    st.markdown("**Don't have values? TRY THESE INPUTS** ")
    st.markdown("119.99200,157.30200,74.99700,0.00784,0.00007,0.00370,0.00554,0.01109,0.04374,0.42600,0.02182,0.03130,0.02971,0.06545,0.02211,21.03300,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654	 --> **output** : present")  
    #getting the input data from the user
    #columns for input fields
    
    #the order must be same as in the data set
    
    
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        parkinson_pred = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        parkinson_pred = [float(x) for x in parkinson_pred]

        parkinson_prediction = parkinsons_model.predict([parkinson_pred])
        
        if parkinson_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)
    


#breast cancer prediction page
if(selected == 'Breast Cancer Prediction'):
    st.title('Breast Cancer prediction using Logistic Regression')
    st.text("This project aims to build a Classification model to detect the presence of breast cancer.")
    st.text("First value Malignant; Second value is Benign")
    #getting the input data from the user
    #columns for input fields
    #the order must be same as in the data set   
    
    col1, col2, col3 = st.columns(3)

    with col1:
        mean_radius = st.selectbox("Mean Radius", [19.69, 13.08])
        
    with col2:
        mean_texture = st.selectbox("Mean texture", [21.25, 15.71])

    with col3:
        mean_perimeter = st.selectbox("Mean perimeter", [130, 85.63])

    with col1:
        mean_area = st.selectbox("Mean area", [1203, 520])

    with col2:
        mean_smoothness = st.selectbox("Mean smoothness", [0.1096, 0.1075])

    with col3:
        mean_compactness = st.selectbox("Mean compactness", [0.1599, 0.127])

    with col1:
        mean_concavity = st.selectbox("Mean concavity", [0.1974, 0.04568])

    with col2:
        mean_concave_points = st.selectbox("Mean concave", [0.1279, 0.0311])

    with col3:
        mean_symmetry = st.selectbox("Mean symmetry", [0.2069, 0.1967])

    with col1:
        mean_fractal_dimensions = st.selectbox("Mean Fractal Dimensions", [0.05999, 0.06811])

    with col2:
        radius_error = st.selectbox("Radius Error", [0.7456, 0.1852])
        
    with col3:
        texture_error = st.selectbox("Texture Error", [0.7869, 0.7477])

    with col1:
        perimeter_error = st.selectbox("Perimeter Error", [4.585, 1.383])

    with col2:
        area_error = st.selectbox("Area error", [94.03, 14.67])

    with col3:
        smoothness_error = st.selectbox("Smoothness error", [0.00615, 0.004097])

    with col1:
        compactnesss_error = st.selectbox("compactness error", [0.04006, 0.01898])

    with col2:
        concavity_error = st.selectbox("Concavity Error", [0.03832, 0.01698])

    with col3:
        concave_points_error = st.selectbox("Concave Points error", [0.02058, 0.00649])

    with col1:
        symmetry_error = st.selectbox("Symmetry error", [0.0225, 0.01678])

    with col2:
        fractal_dimension_error = st.selectbox("Fractal Dimension error", [0.004571, 0.002425])

    with col3:
        worst_radius = st.selectbox("Worst radius", [23.57, 14.5])

    with col1:
        worst_texture = st.selectbox("Worst Texture", [25.53, 20.49])

    with col2:
        worst_preimeter = st.selectbox("Worst Preimeter", [152.5, 96.09])

    with col3:
        worst_area = st.selectbox("Worst area", [1709, 630.5])

    with col1:
        worst_smoothness = st.selectbox("worst smoothness", [0.1444, 0.1312])

    with col2:
        worst_compactness = st.selectbox("worst ccompactness", [0.4245, 0.2776])

    with col3:
        worst_concavity = st.selectbox("Worst Concavity", [0.4504, 0.189])

    with col1:
        worst_concave_points = st.selectbox("Worst concave points", [0.243, 0.07283])

    with col2:
        worst_symmetry = st.selectbox("worst symmetry", [0.3613, 0.3184])

    with col3:
        worst_fractal_dimension = st.selectbox("Worst Fractal dimension", [0.08758, 0.08183])


        
    
    
    #code for prediction
    breast_diagnosis = ''  #empty string to store the end result
    
    #creating a button for prediction
    if st.button('Breast Cancer Test Result'):
        breast_pred = [
            mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
            mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimensions,
            radius_error, texture_error, perimeter_error, area_error, smoothness_error,
            compactnesss_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
            worst_radius, worst_texture, worst_preimeter, worst_area, worst_smoothness,
            worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension]



        #the columns names are in 2 square brackets to tell the model that we are predicting for one data point
        
        breast_pred = [float(x) for x in breast_pred]

        breast_prediction = breast_cancer_model.predict([breast_pred])
        

        if(breast_prediction[0] == 1):
            breast_diagnosis = 'The tumor is benign (Non-cancerous)'
        else:
            breast_diagnosis = 'The tumor is malignant (Cancerous)'

    st.success(breast_diagnosis)
    

st.write("For help, contact: jayshivareddy@gmail.com")









