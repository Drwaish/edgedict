from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        print(file)
        if file:
            filename = secure_filename(file.filename)
            file.save(filename)
            # Do something with the uploaded file, e.g., process it or save it to a database
            return 'File uploaded successfully'
    return render_template('index.html')

if __name__ == "__main__":
    app.run()



import socketio

sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files={
    '/': './public/'
})


@sio.event
def connect(sid, environ):
    print(sid, 'connected')


@sio.event
def disconnect(sid):
    print(sid, 'disconnected')


@sio.event
def sum(sid, data):
    # result = data['numbers'][0] + data['numbers'][1]
    result = 4
    sio.emit('sum_result', {'result': result}, to = sid)