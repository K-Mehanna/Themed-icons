# Libaries to help with the database
import os

# Libraries used 
from flask import Flask, flash, redirect, render_template, request, url_for, session, send_from_directory, current_app, send_file
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.utils import secure_filename
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash

from edge_detection import detect_edges
from theme_icon import theme_icon

# Configure application
app = Flask(__name__)

# Ensure templates (html webpages) are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Specifies the location to save files
UPLOAD_FOLDER = "images"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/create-icon", methods=["POST"])
def create_icon():
    (filename, bg_colour, fg_colour, icon_type) = get_icon_and_form_data()
    print(f'filename: {filename}, fg_colour: {fg_colour}, bg_colour: {bg_colour}, icon_type: {icon_type}')

    folder_path = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
    (soft_edges_mask, harsh_edges_mask) = detect_edges(folder_path, filename)
    themed_icon_filename = theme_icon(folder_path, filename, bg_colour, fg_colour, icon_type, soft_edges_mask, harsh_edges_mask)

    return send_from_directory(folder_path, themed_icon_filename, as_attachment=True)

def create_icon_from_image(filename, bg_colour, fg_colour, icon_type):
    # Call function to create icon
    return filename

def get_icon_and_form_data():
    icon = request.files["file_input"]
    if icon.filename == '':
        flash('Must provide file', 'error')
        print("must include image")
        return redirect(url_for("index"))
    elif not allowed_file(icon.filename):
        flash('Must be a png, jpg or jpeg', 'error')
        print("must be a png, jpg or jpeg")
        return redirect(url_for("index"))
  
    filename = secure_filename(icon.filename)
    icon.save(os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'], filename))

    bg_colour = request.form['bg_colour']
    fg_colour = request.form['fg_colour']
    icon_type = request.form['icon_type']
        
    return (filename, bg_colour, fg_colour, icon_type)

def allowed_file(filename):
    allowedExtensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowedExtensions

if __name__ == "__main__":
    app.run(debug=True)