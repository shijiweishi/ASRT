from flask import Flask,url_for
from gevent import pywsgi

app = Flask(__name__) # 实例化类flask
@app.route('/')
def hello_world():
    return "<h2 style='color:red'>Hello World</h2>"


@app.route('/user/<username>')
def show_user(username):
    return f'我是{username}'


@app.route('/test/')
def atest():
    return url_for('show_user', username='Andy')  # (函数名，参数赋值)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080,debug=True)

