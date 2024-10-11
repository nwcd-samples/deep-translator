from flask import Flask, render_template

# app = Flask(__name__, template_folder='../templates', static_folder='../static')
app = Flask(__name__)
app.config['SECRET_KEY'] = 'deep-speech'


@app.route("/")
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
