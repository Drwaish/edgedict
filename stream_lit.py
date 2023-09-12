import streamlit as st
import time 

# Define the callback function
def button_callback():
    # Perform the desired action
    st.write('Start Speak!')

# Create the button
if st.button('Click me', on_click=button_callback):
    # The callback function will be executed when the button is clicked
    t = st.empty()
    # for i in range(10):
    #     time.sleep(1)
    #     st.text("Hello"+str(i))
    text = "Welcome to the first day... of the rest... of your life"


    for i in range(len(text) + 1):
        t.markdown("## %s..." % text[0:i])
        time.sleep(0.1)