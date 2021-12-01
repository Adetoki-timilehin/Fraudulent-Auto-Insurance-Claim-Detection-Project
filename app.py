from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model_2 = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET"])
def home_page():
    return render_template("index.html")

@app.route("/predict", methods = ["GET","POST"])
def prediction():
        user_input= []
        names = ["a","b","f","g","h","i","m","n","o","p","q","r","s","t","u", "v"]
        for y in names:
            user_input_temp = request.args.get(y)
            user_input_per = int(user_input_temp)
            user_input.append(user_input_per)    
        user_input_2 = [user_input]
        print(user_input_2)
        prediction = model_2.predict_proba(user_input_2)

        if prediction[:,1] > 0.7:
            return render_template("after.html", dat = "This is likely to be fraudulent")
        elif prediction[:,1] < 0.3:
            return render_template("after.html", dat = "This is very unlikely to be fraudulent")
        else:
            return render_template("after.html", dat = "This is not a decidable case")
       



if __name__ == "__main__":
    app.run(port = 2000, debug = True)