import gradio as gr
import pandas as pd
import numpy as np

# D1 imports
from D1dataloader import get_data as d1_get_data
from D1models import get_classifiers as d1_get_classifiers

# D2 imports
from D2dataloader import load_data as d2_load_data
from D2Preprocessing import preprocess_data as d2_preprocess_data
from D2Models import train_and_evaluate as d2_train_and_evaluate


D1_DATA_PATH = "data/adult.csv"
D1_X_train, D1_X_test, D1_y_train, D1_y_test = d1_get_data(D1_DATA_PATH, for_clustering=False)
D1_model = d1_get_classifiers()['RandomForest']
D1_model.fit(D1_X_train, D1_y_train)
D1_columns = D1_X_train.columns

D1_WORKCLASS = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked']
D1_MARITAL = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent']
D1_OCCUPATION = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
D1_RELATIONSHIP = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
D1_RACE = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
D1_SEX = ['Male', 'Female']
D1_COUNTRY = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']

def d1_predict(
    age, workclass, education_num, marital_status, occupation, relationship,
    race, sex, capital_gain, capital_loss, hours_per_week, native_country
):
    user_dict = {
        'age': age,
        'workclass': workclass,
        'education-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }
    user_df = pd.DataFrame([user_dict])
    for col in D1_columns:
        if col not in user_df.columns:
            user_df[col] = 0
    user_df = user_df[D1_columns]
    pred = D1_model.predict(user_df)[0]
    return " Income >50K" if pred == 1 else " Income <=50K"


D2_df = d2_load_data()
D2_df_processed = d2_preprocess_data(D2_df)
D2_X = D2_df_processed.drop('Survived', axis=1)
D2_y = D2_df_processed['Survived']
from sklearn.model_selection import train_test_split
D2_X_train, D2_X_test, D2_y_train, D2_y_test = train_test_split(D2_X, D2_y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
D2_model = LogisticRegression(max_iter=200)
D2_model.fit(D2_X_train, D2_y_train)
D2_columns = D2_X_train.columns

def d2_predict(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    user_dict = {
        "Pclass": Pclass,
        "Sex": 0 if Sex == "male" else 1,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked": {"S":0,"C":1,"Q":2}[Embarked]
    }
    user_df = pd.DataFrame([user_dict])
    pred = D2_model.predict(user_df)[0]
    return " Survived" if pred == 1 else " Did not Survive"


modern_css = """
body { background: linear-gradient(120deg, #e0eafc 0%, #cfdef3 100%); }
.gradio-container { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; }
.gradio-title { color: #283e51; font-size: 2.5rem; font-weight: 700; margin: 0 0 1.5rem 0; text-align:center; letter-spacing: -1px; }
.gradio-description { color: #6b7280; font-size: 1.15rem; margin-bottom: 2rem; text-align:center;}
.gradio-tabs { margin-bottom: 2rem; }
.gradio-input label { font-weight: 600; color: #344e68; }
.gradio-interface { border-radius: 18px; box-shadow: 0 6px 32px rgba(44,62,80,.09); background: #fff; padding: 1.5rem 2.5rem; }
.gradio-block { padding-top: 1.2rem; padding-bottom: 1.2rem; }
.gr-tabitem { border-radius: 14px 14px 0 0; background: #f0f4fa; }
input, select, textarea { border-radius: 8px !important; border: 1px solid #dedede !important; font-size: 1.08rem !important; }
.gr-button { background: linear-gradient(90deg, #43cea2, #185a9d); color: #fff; border: none; font-weight: 600; border-radius: 8px; }
.gr-button:hover { background: linear-gradient(90deg, #185a9d, #43cea2); }
.gr-textbox, .gr-textbox textarea { background: #f7fafd !important; border-radius: 8px; border: 1px solid #e3e8ee; color: #283e51; }
.gradio-output label { color: #185a9d; font-weight: 600;}
::-webkit-scrollbar-thumb { background: #c2d3e9 !important; border-radius: 8px; }
::-webkit-scrollbar { background: #f2f5fa !important; width: 9px; }
"""

with gr.Blocks(css=modern_css) as demo:
    gr.Markdown("<div class='gradio-title'> Machine Learning Model Playground</div>")
    gr.Markdown("<div class='gradio-description'>Interact with two modern ML models: <b>Adult Income</b> & <b>Titanic Survival</b>. Enter data below and see predictions instantly.</div>")
    with gr.Tabs():
        with gr.TabItem("Adult Income Prediction (D1)"):
            gr.Markdown("####  Predict whether a person earns <b>&gt;50K</b> or <b>&lt;=50K</b>")
            d1_inputs = [
                gr.Number(label="Age", value=35, elem_classes=["gradio-input"]),
                gr.Dropdown(D1_WORKCLASS, label="Workclass", elem_classes=["gradio-input"]),
                gr.Number(label="Education Num", value=10, elem_classes=["gradio-input"]),
                gr.Dropdown(D1_MARITAL, label="Marital Status", elem_classes=["gradio-input"]),
                gr.Dropdown(D1_OCCUPATION, label="Occupation", elem_classes=["gradio-input"]),
                gr.Dropdown(D1_RELATIONSHIP, label="Relationship", elem_classes=["gradio-input"]),
                gr.Dropdown(D1_RACE, label="Race", elem_classes=["gradio-input"]),
                gr.Dropdown(D1_SEX, label="Sex", elem_classes=["gradio-input"]),
                gr.Number(label="Capital Gain", value=0, elem_classes=["gradio-input"]),
                gr.Number(label="Capital Loss", value=0, elem_classes=["gradio-input"]),
                gr.Number(label="Hours per Week", value=40, elem_classes=["gradio-input"]),
                gr.Dropdown(D1_COUNTRY, label="Native Country", value="United-States", elem_classes=["gradio-input"]),
            ]
            gr.Interface(
                fn=d1_predict,
                inputs=d1_inputs,
                outputs=gr.Textbox(label="Prediction", elem_classes=["gradio-output"]),
                live=False,
                allow_flagging='never',
                theme='default'
            ).render()

        with gr.TabItem("Titanic Survival Prediction (D2)"):
            gr.Markdown("####  Predict whether a passenger <b>survived</b> the Titanic or not")
            d2_inputs = [
                gr.Radio([1, 2, 3], label="Passenger Class (Pclass)", value=3, elem_classes=["gradio-input"]),
                gr.Radio(["male", "female"], label="Sex", value="male", elem_classes=["gradio-input"]),
                gr.Number(label="Age", value=30, elem_classes=["gradio-input"]),
                gr.Number(label="Siblings/Spouses (SibSp)", value=0, elem_classes=["gradio-input"]),
                gr.Number(label="Parents/Children (Parch)", value=0, elem_classes=["gradio-input"]),
                gr.Number(label="Fare", value=32, elem_classes=["gradio-input"]),
                gr.Radio(["S", "C", "Q"], label="Port of Embarkation (Embarked)", value="S", elem_classes=["gradio-input"]),
            ]
            gr.Interface(
                fn=d2_predict,
                inputs=d2_inputs,
                outputs=gr.Textbox(label="Prediction", elem_classes=["gradio-output"]),
                live=False,
                allow_flagging='never',
                theme='default'
            ).render()

if __name__ == "__main__":
    demo.launch()