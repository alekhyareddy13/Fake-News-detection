from flask import Flask ,request,render_template
import pickle
import pandas as pd

df = pd.read_csv("news.csv")
vector=pickle.load(open("vectorizer.pkl",'rb'))
model=pickle.load(open("finalized_model.pkl",'rb'))


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction",methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        input_text = str(request.form['news'])  # Input text provided by the user
        filtered_df = df[df['title'].str.contains(input_text, case=False)]  # Filter the DataFrame to find news containing the input text

        if not filtered_df.empty:
            headline = filtered_df.iloc[0]['title']  # Get the headline of the first matching news
            predict = model.predict(vector.transform([headline]))
            print(predict)

            return render_template("prediction.html",
                                   headline=headline,
                                   prediction_text="{}".format( predict[0]))
        else:
            return render_template("prediction.html", prediction_text="No matching news found")

    else:
        return render_template("prediction.html")
        


if __name__=='__main__':
    app.run()