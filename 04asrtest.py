import os
from ASR import recongnize


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from flask import Flask,url_for

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

@app.route('/profile')
def profile():  # put application's code here
    return '个人中心'

@app.route('/asr')
def asr1():
    r1,r2 = recongnize()

    print(r1,r2)
    return '拼音:{},文字:{}'.format(r1,r2)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8008, threaded=False)

