import streamlit as st
import pandas as pd
from util_function import run_regression, run_linear, clean_data, get_data_for_component, get_data_without_rotation, get_data_with_250

def categorize(value):
    if 0 <= value <= 25:
        return 0
    elif 25 < value <= 40:
        return 1
    else:
        return 2
    
def categorize_image(value):
    if 0 <= value <= 125:
        return 0
    elif 125 < value <= 250:
        return 1
    else:
        return 2

st.title('Model Comparison')

st.sidebar.header('Image Settings')

options = st.sidebar.multiselect(
    "Select independent variables:",
    ["len_of_boundry","len_of_boundry_per_100px", "disjoint_image", "neighbours"]
)

only_one_component = st.sidebar.checkbox("Test on 1 component")

rotation = st.sidebar.checkbox("Without Rotated Image")

multiple_tile_size  = st.sidebar.checkbox("Multiple Tile size")

single_model  = st.sidebar.checkbox("Single Model")

#remove_all_white_image = st.sidebar.checkbox("Remove all white image")
remove_all_white_image = True
remove_all_white_image = st.sidebar.radio('Pick one : ', ['With White Image','Remove white image with No neighbour','Remove all white image'])

st.sidebar.header('Model Settings')

a = st.sidebar.radio('Pick one : ', ['Categorical','Linear'])

dropout_percent = 0
if a == 'Categorical':
    batch_norm = st.sidebar.checkbox("Batch Normalization")
    drop_out = st.sidebar.checkbox("Use dropout ?")
    if drop_out is True:
        dropout_percent = st.sidebar.slider('Dropout percent', 1,100, step=1)
        dropout_percent = dropout_percent/100

#batch_norm = False
#dropout_percent = 0
#a = 'Categorical'
layers = st.sidebar.slider('Number of layers', 1, 5, step=1)
num_of_cell = st.sidebar.slider('Number of filters in each layer', 16, 128, step=16)
learning_rate = st.sidebar.number_input(label="Learning rate", format="%f")
batch_size = st.sidebar.slider('Batch Size', 16, 64, step=16)
epochs = st.sidebar.slider('Number of iteration', 5, 50, step=5)

optimizer = st.sidebar.selectbox("Optimizer", ('adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam'))

if a == 'Categorical':
    if multiple_tile_size:
        cols = ['name','len_of_boundry','len_of_boundry_per_100px', 'disjoint_image', 'neighbours','density_category','num_triangles','triangle_per_100px']
        y = 'density_category'
    else:
        cols = ['name','len_of_boundry', 'len_of_boundry_per_100px','disjoint_image', 'neighbours','image_category','num_triangles','triangle_per_100px']
        y = 'image_category'
else:
    cols = ['name','len_of_boundry_per_100px', 'disjoint_image', 'neighbours','num_triangles']

st.text('Sample Data')
data = pd.read_csv('./output/training_data/train.csv', usecols=cols)
data = data[cols]
data = clean_data(remove_all_white_image, data)
if only_one_component:
    data = get_data_for_component('microstrip_coupler', data)
if rotation:
    data = get_data_without_rotation(data)
if not multiple_tile_size:
    data = get_data_with_250(data)
    data['image_category'] = data['num_triangles'].apply(categorize_image)
else:
    data['density_category'] = data['triangle_per_100px'].apply(categorize)
st.dataframe(data=data, use_container_width=True, height=200)

st.text(f'Number of records : {data.shape[0]}')

if st.button('Run Model'):
    if a == 'Categorical':
        test_loss, test_accuracy, fig, confusion_mat = run_regression(remove_all_white_image,only_one_component, y, options, layers,num_of_cell, 
                                                       epochs, batch_norm, dropout_percent, learning_rate, batch_size, optimizer, data, single_model)
        
        st.text(f'Test accuracy: {test_accuracy}')
        st.text(f'Test loss: {test_loss}')
        st.pyplot(fig[0])
        st.pyplot(fig[1])
        st.pyplot(confusion_mat)
        
    else:
        test_loss, test_mae, fig = run_linear(remove_all_white_image, options, layers, num_of_cell, epochs)
        st.pyplot(fig[0])
        st.pyplot(fig[1])
        # Print the test loss and MAE
        st.text(f'Test Loss: {test_loss}')
        st.text(f'Test MAE: {test_mae}')

