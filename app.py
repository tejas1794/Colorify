# Imports
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import argparse
import matplotlib.pyplot as plt
from model import *

# Setup Flask app with template_folder path as "pages/"
app = Flask(__name__, template_folder="pages")
app.secret_key = "8sd9fh39fgh398fh3"
UPLOAD_FOLDER = 'static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def colorize(filepath):
    """Takes an image page, loads model and creates prediction image"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', type=str, default=filepath)
    opt = parser.parse_args()
    # Loading ML model
    model = load_trained_model(pretrained=True).eval()
    # Converting image to lab space and resizing to 256x256
    (orig, resized) = preprocess(load_image(opt.img_path), dims=(256, 256))
    # Using the 256x256 lab space output, convert and resize to match original image and rgb space and save image as
    # result.png
    plt.imsave('static/result.png', postprocess(orig, model(resized).cpu()))


def predict(imagepath):
    """Takes a valid 64bit image file, preprocessed and returns the final prediction"""
    colorize(imagepath)


def allowed_file(filename):
    """Take a filename and returns whether the file is of correct format"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Default route handling
@app.route('/', methods=['GET', 'POST'])
def initialize():
    """Handles home page and post calls retrieving the image query"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # print('upload_image filename: ' + filename)
            flash('Image successfully uploaded - see the conversion below:')
            predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('index.html', filename=filename,
                                   prediction='result.png')  # prediction=str(round(result[0]/100)))
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
    return render_template('index.html')


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


# Start app
if __name__ == '__main__':
    app.run(debug=True)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
