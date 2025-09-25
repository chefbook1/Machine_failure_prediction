import streamlit as st
import joblib
import pandas as pd

# Load the model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('machine_failure_rf_model.pkl')
    scaler = joblib.load('machine_failure_scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Streamlit UI
st.title("Machine Failure Prediction 🏭")
 
# Input form
with st.form("prediction_form"):
    st.subheader("Machine Sensor Data Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        footfall = st.number_input("Footfall (people/objects passing by)", min_value=0, value=0)
        temp_mode = st.number_input("Temperature Mode (machine setting)", min_value=0, max_value=7, value=0)
        aq = st.number_input("Air Quality Index (AQ)", min_value=0, value=0)
        uss = st.number_input("Ultrasonic Sensor (USS) - proximity", min_value=0, value=0)
        
    with col2:
        cs = st.number_input("Current Sensor (CS) - electrical current", min_value=0, value=0)
        voc = st.number_input("Volatile Organic Compounds (VOC) level", min_value=0, value=0)
        rp = st.number_input("Rotational Position (RP) - RPM", min_value=0, value=0)
        ip = st.number_input("Input Pressure (IP)", min_value=0, value=0)
    
    temperature = st.number_input("Operating Temperature", min_value=0, value=0)
    
    submitted = st.form_submit_button("Predict Failure")

# Prediction logic
if submitted:
    # Prepare input data
    input_data = pd.DataFrame([[footfall, temp_mode, aq, uss, cs, voc, rp, ip, temperature]],
                            columns=['footfall', 'tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature'])
    
    # Scale the input
    try:
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)
        
        # Display result
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"🚨 Machine Failure Predicted! (Probability: {prediction_proba[0][1]:.2%})")
            st.warning("Recommended Action: Schedule maintenance immediately")
        else:
            st.success(f"✅ No Failure Predicted (Probability: {prediction_proba[0][0]:.2%})")
            st.info("Machine operating within normal parameters")
            
        # Show input summary
        with st.expander("View Input Summary"):
            st.dataframe(input_data)
            
    except Exception as e:
        st.error(f"Error in processing: {str(e)}")
        st.warning("Please check your input values and try again")