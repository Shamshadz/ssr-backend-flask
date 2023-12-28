# Import necessary libraries
from flask import Flask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from flask import Flask, request, jsonify
import io
from flask import send_file
import sklearn
import pickle
import os
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

def LoadData(simout_path,label_path):
  df1 = pd.read_csv(simout_path)
  df1['SSR'] = 0
  df2 = pd.read_csv(label_path)
  return df1,df2

def LabelData(dataset,time_range):
  for _, row in time_range.iterrows():
    ll = row['ROILimits_1']
    ul = row['ROILimits_2']
    dataset.loc[(dataset['Time'] >= ll) & (dataset['Time'] <= ul), 'SSR'] = 1
  return dataset

def file_uploaded():
  simout_path = './uploads/simout10_60_15.csv'
  label_path = './uploads/labeled_60_15.csv'

  df1,df2 = LoadData(simout_path,label_path)
  df1 = LabelData(df1,df2)

  # # Assuming you have a DataFrame 'df' with features and target variable
  X = df1.drop('SSR', axis=1)
  y = df1['SSR']

  return df1, X, y


def PlotI(dataset):
  plt.figure(figsize=(10, 6))
  plt.plot(dataset.Time,dataset.Ia, color='r')
  plt.xlim([2,5])
  plt.xlabel('Time')
  plt.ylabel('Ia')
  plt.show()

  # Save the plot as an image
  img_buffer = io.BytesIO()
  plt.savefig(img_buffer, format='png')
  img_buffer.seek(0)

  # Return the image buffer
  return img_buffer

def VisualizeSSR(dataset,df1):
  ssr_1_data = dataset[dataset['SSR'] == 1]
  ssr_0_data = dataset[dataset['SSR'] == 0]
  plt.figure(figsize=(10, 6))
  plt.plot(ssr_1_data['Time'], ssr_1_data['Ia'], label='Ia (ssr=1)', color='k')
  plt.plot(ssr_0_data['Time'], ssr_0_data['Ia'], label='Ia (ssr=0)', color='gray')
  plt.xlabel('Time')
  plt.ylabel('Ia')
  time = df1.loc[df1['SSR'] == 1,'Time']
  r1,r2 = time.iloc[0],time.iloc[int(time.shape[0]/4)]
  plt.axvspan(r1, r2, facecolor='red',edgecolor='blue',linewidth=10, alpha=0.5)
  plt.title('Time vs Ia with SSR Marked')
  plt.legend()
  plt.show()

  # Save the plot as an image
  img_buffer = io.BytesIO()
  plt.savefig(img_buffer, format='png')
  img_buffer.seek(0)

  # Return the image buffer
  return img_buffer


def combine_images_vertically(img1, img2):
    # Open the images using Pillow
    image1 = Image.open(img1)
    image2 = Image.open(img2)

    # Assuming both images have the same width
    width = image1.width

    # Create a new image with double the height to accommodate both images one below the other
    composite_image = Image.new('RGB', (width, image1.height + image2.height))
    
    # Paste the individual images onto the composite image
    composite_image.paste(image1, (0, 0))
    composite_image.paste(image2, (0, image1.height))

    # Save the composite image to a buffer
    composite_buffer = io.BytesIO()
    composite_image.save(composite_buffer, format='PNG')
    composite_buffer.seek(0)

    return composite_buffer

@app.route("/plot")
def plot():
    df1, X, y = file_uploaded()
    
    # Get the image buffers
    img_buffer0 = PlotI(df1)
    img_buffer1 = VisualizeSSR(df1, df1)

    # Combine the images into a single composite image (vertically)
    composite_buffer = combine_images_vertically(img_buffer0, img_buffer1)

    # Send the composite image in the response
    return send_file(composite_buffer, mimetype='image/png')


@app.route("/score")
def score():
    try:
        # Use joblib to load the model instead of pickle
        loaded_model = pickle.load(open('./decisiontree.sav', 'rb'))
        df1, X, y = file_uploaded()
        result = loaded_model.score(X, y)
        score = result*100
        return jsonify({'score': score})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/")
def hello():
  return "Hello World!"

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    try:
        # Check if the POST request has file parts
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({'error': 'Both files must be provided'})

        file1 = request.files['file1']
        file2 = request.files['file2']

        # Check if the files are provided
        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'Both files must be selected'})

        # Check if the files have allowed extensions (optional)
        allowed_extensions = {'csv'}
        for file in [file1, file2]:
            if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                return jsonify({'error': 'Invalid file extension'})

        # Save the CSV files locally
        upload_path = './uploads'
        os.makedirs(upload_path, exist_ok=True)  # Create the 'uploads' directory if it doesn't exist
        file1_path = os.path.join(upload_path, file1.filename)
        file2_path = os.path.join(upload_path, file2.filename)
        file1.save(file1_path)
        file2.save(file2_path)

        # Read the CSV files into DataFrames
        df_uploaded1 = pd.read_csv(file1_path)
        df_uploaded2 = pd.read_csv(file2_path)

        # Return a success message or any other response
        return jsonify({'message': 'Files uploaded and processed successfully'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
