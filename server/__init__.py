from datetime import datetime
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
# CORS(app)

app.config['SECRET_KEY'] = '31f214ac7307802de7160100ec7a549b'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'


from server import controllers