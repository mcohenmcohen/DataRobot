from flask import Flask, session, redirect, url_for, escape, request
app = Flask(__name__)
app.debug=True

@app.route('/')
def indx():
    return 'What up!'

@app.route('/hello', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'

@app.route("/test", methods=['GET', 'POST'])
def test():
    name_of_slider = request.form["name_of_slider"]
    return name_of_slider

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['nm']
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))

def shutdown_server():
    '''
    Can find the process with ps -fA | grep python
    '''
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

# @app.route('/shutdown', methods=['POST'])
@app.route('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

if __name__ == '__main__':
   app.run()
