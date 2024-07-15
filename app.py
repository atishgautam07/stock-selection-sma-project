from flask import Flask, render_template, request, jsonify
import os 
from stockSelProj.pipeline.predict import PredictionPipeline

ml_uri =  "http://34.93.200.115:5000"    #"http://127.0.0.1:5000"
hpo_exp_rf = "random-forest-hyperParamTune v1"
hpo_exp_xgb = "xgBoost-hyperParamTune v1"
model_name_xgb = "best-model-xgb"
model_name_rf = "best-model-rf"
exp_name = "best-models v1"
prediction_name = "pred_xgp_rf_best"

# app = Flask(__name__) # initializing a flask app

app = Flask('stock-selection')

@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n")
    return "Training Successful!" 

@app.route('/predict', methods=['GET'])
def predict_endpoint():
    # ride = request.get_json()

    trained = PredictionPipeline(ml_uri, exp_name, model_name_rf, model_name_xgb)
    trained.load()

    prediction_name='pred_xgp_rf_best'
    trained.predict(pred_name=prediction_name)
    
    COLUMNS = ['Ticker', 'Adj Close','Date',prediction_name, prediction_name+'_rank']
    result_df = trained.df_full[(trained.df_full[f'{prediction_name}_rank']<=10) & 
                                (trained.df_full['Date'] == trained.df_full['Date'].max())].sort_values(by=prediction_name)[COLUMNS]

    result = result_df.to_dict()
    print (result)
    return jsonify(result)


if __name__ == "__main__":
    # training()
    # predict_endpoint()
    app.run(debug=True, host='0.0.0.0', port=9696)