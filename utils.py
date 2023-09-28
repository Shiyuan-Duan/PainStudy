import numpy as np
import pandas as pd

def get_data(subject):
    anno_file = pd.read_csv(r'./FINAL times lab baseline for ML 9_5_23.csv', encoding='utf-8')
    anno_file.set_index('Subject N', inplace=True)
    # anno_file.loc['S1']['baseline HR']
    file_path = f'.\\Data_Summary\\{subject}_Summary.csv'
    cols = ['L_Active', 'itp_Po', 'itp_Temp', 'itp_HR', 'itp_RR']
    data = pd.read_csv(file_path)
    data = data.set_index('time(s)')
    data = data[cols]
    data['label'] = np.zeros(len(data))


    anno = anno_file.loc[f'S{subject}']

    base_hr = anno['baseline HR']
    base_rr = anno['baseline RR']
    start = anno['Sensor Lab time (s)']
    end = anno['Sensor lab stop 9s)']
    data['itp_HR'] -= base_hr
    data['itp_RR'] -= base_rr
    data.loc[start:end, 'label'] = 1
    data = data.ffill().bfill()
    # data = (data - data.mean())/(data.std() + 1e-10)
    # print(data.loc[start:end])
    return data.values

