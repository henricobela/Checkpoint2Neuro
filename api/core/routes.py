from flask import Flask, Blueprint, render_template, request, jsonify, redirect, url_for
from api.core.enums.http_status import  HttpStatus
from api.core.response.types import PayloadResponse
from api.core.utils import *
import json, sqlite3, requests
import numpy as np
from datetime import datetime
import pandas as pd
import time


bp = Blueprint('core',__name__)
bp_v1 = Blueprint('status',__name__)


def get_data():
    get_df = TksRequest()
    df = get_df.get_data_response()
    return df


# Endpoint inicial mostrando membros da equipe
@bp.route('/')
def hello():
    return render_template('index.html')


# Endpoint de retorno do modelo
@bp.route('/return_model_response', methods = ['GET'])
def return_model_response():
    start_time = time.time()

    data = request.get_json()
    df = pd.DataFrame(data, index=[0])

    model = Predict_model()
    preds = model.real_predict(df)
    text = " ".join(str(i) for i in preds)

    end_time = time.time()
    processing_time = end_time - start_time

    return jsonify({'Temperature Predict': text,
                    'Estimated Time': processing_time})


# Endpoint de visualização dos dados provenientes das APIs
@bp.route('/see_all_data', methods=['POST'])
def see_all_data():
    start_time = time.time()

    df = get_data()

    end_time = time.time()
    processing_time = end_time - start_time
    
    return jsonify({'message': df.to_json(orient='records', lines=True),
                    'processing_time_seconds': processing_time})



