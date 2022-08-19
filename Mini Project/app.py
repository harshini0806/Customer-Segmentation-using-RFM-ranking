import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.cluster import KMeans
from fcmeans import FCM
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
app = Flask(__name__)
r = []
f = []
m = []
algo= []
X = []
@app.route("/")
def upload():
   return render_template('upload.html')

app.config.from_object(__name__)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save("D:\\Mini project\\file.csv")
      data = pd.read_csv("D:\\Mini project\\file.csv", index_col=0)
      return render_template('display.html',  tables=[data.head().to_html(classes='data')], titles=data.columns.values)

# @app.route("/input", methods=["GET","POST"])
# def input():
#     return render_template('input.html')

@app.route("/select_algo", methods=["GET", "POST"])
def select_algo():
    models = [KMeans(), FCM()]
    if request.method == "POST":
        algo.append(request.form.get("algo"))
    return render_template("select_algo.html",models=models)

@app.route("/inputval", methods =["GET", "POST"])
def submit_values():
    if request.method == 'POST':
        r.append(request.form.get("valuer"))
        f.append(request.form.get("valuef"))
        m.append(request.form.get("valuem"))
        print(r, f, m)
    return render_template("input.html")

@app.route("/select_features", methods=["GET", "POST"])
def select_features():
    data = pd.read_csv("D:\\Mini project\\file.csv", index_col=0)
    ids= data.columns
    if request.method == "POST":
        X.append(request.form.getlist("ids"))
        print(X)
    return render_template('select_features.html', ids= ids)


def algores(algo):
    alg = None
    if algo[0] == "KMeans()":
        alg = KMeans(n_clusters=5)
    elif algo[0] == "FCM()":
        alg = FCM(n_clusters=5)
    data = pd.read_csv("D:\\Mini project\\file.csv", index_col=0)
    x = X[-1]
    
    alg.fit(data[x])
    x_2 = [int(r[-1]), int(f[-1]), int(m[-1])]
    x_22 = np.reshape(x_2, (1, -1))
    preds = alg.predict(x_22)
    return preds[0]  


@app.route("/result", methods=["GET", "POST"])
def result():
    result = algores(algo)
    if result == 5:
        res = "Very Good Customer"
    elif result == 4:
        res= "Good Customer"
    elif result==3:
        res="Average Customer"
    elif result==2:
        res="Below Average Customer"
    elif result==1:
        res="Poor Customer"
    elif result==0:
        res="Very Poor Customer"
    return render_template("result.html", result=result, res = res)


if __name__ == '__main__':
    app.run(debug = True)
