#!/usr/bin/env python
# coding: utf-8

# get_ipython().system('pip freeze | grep scikit-learn')
# get_ipython().system('python -V')

import os
import sys
import pickle
import pandas as pd


categorical = ['PULocationID', 'DOLocationID']
def read_data(filename, year, month):
    print (f'reading file - {filename}')
    df = pd.read_parquet(filename)
    print (df.shape)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    return df

def predict_data(dftmp, mdPth):
    print (f"predicting using model from {mdPth}")
    with open(mdPth, 'rb') as f_in:
        dv, model = pickle.load(f_in)

    dicts = dftmp[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    prdStd = y_pred.std()
    prdMean = y_pred.mean()
    print (f"Standard deviation of preds - {prdStd}")
    print (f"Mean of preds - {prdMean}")
    return y_pred

def prep_result_df(dftmp, pred):
    print ("creating results dataframe...")
    df_res = pd.DataFrame({
        'ride_id': dftmp['ride_id'],
        'predicted_duration': pred
    })
    return df_res

def write_result(dfRes, opPath):
    print (f'writing results to - {opPath}')
    dfRes.to_parquet(
        opPath,
        engine='pyarrow',
        compression=None,
        index=False
    )
    sz = os.path.getsize(opPath) / (1024*1024)
    print (f'df_results file-size - {sz}')
    

def run():
    year = int(sys.argv[1]) # 2023
    month = int(sys.argv[2]) # 3
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    modelPath = 'model.bin'
    output_file = f'yellow_{year:04d}-{month:02d}.parquet'


    df = read_data(filename, year, month)
    pred_duration = predict_data(df, modelPath)
    df_result = prep_result_df(df, pred_duration)
    write_result(df_result, output_file)


if __name__ == "__main__":
    run()




