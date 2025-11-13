from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

with open("LE.pkl","rb") as f:
    LE = pickle.load(f)
with open("LR.pkl","rb") as f:
    LR = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction",methods=["POST"])
def prediction():
    sepLen = float(request.form["sepLen"])
    sepWid = float(request.form["sepWid"])
    petLen = float(request.form["petLen"])
    petWid = float(request.form["petWid"])

    pred = LR.predict([[sepLen,sepWid,petLen,petWid]])
    pred_class = int(round(pred[0]))
    flower_name = LE.inverse_transform([pred_class])[0]

    return render_template("index.html",result=flower_name)

if __name__ == "__main__":
    app.run("0.0.0.0",port=5000)