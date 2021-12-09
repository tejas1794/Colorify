# Imports
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import random
import argparse
import matplotlib.pyplot as plt

from colorizers import *

# Load Model
# model = load_model("headcount_model.h5")

# Setup Flask app with template_folder path as "pages/"
app = Flask(__name__, template_folder="pages")
app.secret_key = "8sd9fh39fgh398fh3"
UPLOAD_FOLDER = 'static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def colorize(filepath):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', type=str, default=filepath)
    parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
    parser.add_argument('-o', '--save_prefix', type=str, default='saved',
                        help='will save into this file with {result.png, siggraph17.png} suffixes')
    opt = parser.parse_args()

    # load colorizers
    # colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    if (opt.use_gpu):
        # colorizer_eccv16.cuda()
        colorizer_siggraph17.cuda()

    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    img = load_img(opt.img_path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    if (opt.use_gpu):
        tens_l_rs = tens_l_rs.cuda()

    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    # out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    # plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)
    plt.imsave('static/result.png', out_img_siggraph17)


def predict(imagepath):
    """Takes a valid 64bit image file, preprocessed and returns the final prediction"""
    # return [random.randint(2100, 4800)]
    # img = image.load_img(imagepath, target_size=(224, 224))
    # img = image.img_to_array(img)
    # x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    # prediction = model.predict(x)
    # return np.concatenate(prediction)
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
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Start app
if __name__ == '__main__':
    app.run(debug=True)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
