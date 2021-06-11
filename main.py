import os
from app import app
import urllib.request
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template

from TryCide import RunImage

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','webp'])


@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('files[]')
    file_names = []
    OriginalSizeFlag = request.form.get('hello')
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            print(filename)

            RunImage(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename, OriginalSizeFlag)
    # else:
    #	flash('Allowed image types are -> png, jpg, jpeg, gif')
    #	return redirect(request.url)

    return render_template('upload.html', filenames=file_names)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='Outputimages/' + filename), code=301)


# return redirect(url_for('static', filename='Outputimages/' + filename), code=301)
# OriginalSizeFlag = request.form.get('hello')
# RunImage(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename, OriginalSizeFlag)
if __name__ == "__main__":
    app.run(host='0.0.0.0')