from flask import Flask, render_template, Response, request, session, redirect
import os

app = Flask(__name__)
app.secret_key = "super secret key"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST', 'GET'])
def graph_show():
    text_input = request.form['textInput']
    print(text_input)
       
    has_relation = False
    relation = request.files['myfile']
    print(relation)
    if relation.filename != '':
        relation.save(os.path.join("upload", relation.filename))
        has_relation = True

    noun = 'NONE'
    if request.form.get('enttype2') != None:
        noun = request.form['enttype2']
        print(noun)
    
    is_relation = 'NONE'
    if has_relation:
        is_relation = 'NOT_NONE'

    os.system('python make_kg_graph.py' + ' "' + text_input + '"' + ' ' + noun + ' ' + is_relation)

    return render_template('graph_show.html')

if __name__=='__main__':
    app.debug = True
    app.run(host = "0.0.0.0", port = 5000)

#kill -9 `lsof -i:5001 -t`
#ctrl + c