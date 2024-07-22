import pandas as pd
import matplotlib.pyplot as plt
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
import requests
import json
import os
from threading import Thread
from tkinter import filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from celluloid import Camera
import pickle
 
is_running = False
file_index = 10
#file_name = 'readdata10.csv'pi

#функция остановки записи
def stop_data_process():
    global is_running
    is_running = False
    global file_index 
    file_index += 1
    messagebox.showinfo("Информация", "Запись данных остановлена.")
#функция считывает названия параметров из файла и приводит в вид для считывания
def process_excel_headers(excel_file):
  df = pd.read_csv(excel_file)
  fileheaders = list(df.columns)
  fileheaders  = [header.lower() for header in fileheaders]
  headerskey = fileheaders[0:14]
  updated_list = [ item for item in headerskey]
  jsonstring = str(updated_list)
  jsonstring = jsonstring.replace("'", '"')    
  return df, fileheaders, jsonstring

excel_file = 'data_8_without_duplicates.csv'
dk, fileheaders, jsonstring = process_excel_headers(excel_file)


#функции считывания переменных и создания переменных
vardict = {}
def fetch_dataz(url, prt_id, headers, string):
    global vardict
    prtData_GET = requests.get(url + '/JIS/prtData/' + prt_id, headers=headers)
    if prtData_GET.status_code == 200:
        resp = json.loads(prtData_GET.text)
        if not vardict:
             vardict = resp
        state = list(resp['variables'].values())
        print(state)
        return state
    return None

def create_new_csv_filez(file_index, string):
    global file
    file_name = f'readdata{file_index}.csv'
    column_names = json.loads(string)  # Получаем список столбцов из JSON    
    # Добавляем 'DateTime' в конец списка столбцов
    column_names.append('DateTime')  
    ds = pd.DataFrame(columns=column_names)
    ds.to_csv(file_name, index=False)
    print(f"Created new file: {file_name}")
    return file_name


def writerz(url, headers, string):
    column_names = json.loads(string)
    prtData_POST = requests.post(url + '/JIS/prtData',
                                 data=json.dumps({"variables": json.loads(string)}),
                                 headers=headers)
    if prtData_POST.status_code == 201:
        prt_id = prtData_POST.text.strip()
        print(f"prt_id: {prt_id}")

        global file_index
        file_index = file_index
        global file_name
        #file_name = file_name
        file_name = create_new_csv_filez(file_index, string)
        buffer = []

        while is_running:

            data = fetch_dataz(url, prt_id, headers,string)
            if data:
                buffer.append(data)
# Запись данных в текущий CSV файл
            first_14_columns = column_names[:14]  

            ds = pd.DataFrame(buffer, columns=first_14_columns)  
            ds['DateTime'] = pd.to_datetime('now', format='%Y-%m-%d %H:%M:%S') 
            ds.to_csv(file_name, mode='a', header=False, index=False)
            time.sleep(0.7)
            print("Uploaded successfully")
# Проверка размера файла и создание нового, если размер превышает 10 МБ            
            if os.path.getsize(file_name) > 10 * 1024 * 1024:  # 10 MB
                    file_index += 1
                    file_name = create_new_csv_filez(file_index, string)                
# Читска буфера
            buffer = []
#функция начала записи


def start_data_process():
    global is_running
    is_running = True
    url = 'http://10.176.1.43:8190'
    auth_POST = requests.post(url + '/JIS/users/login',
                             data='{"username": "instructor", "password": ""}',
                             headers={'Content-type': 'application/json'})
    API_TOKEN = json.loads(auth_POST.text)['accessToken']
    print(API_TOKEN)
    headers = {'Authorization': f'Bearer {API_TOKEN}',
           'Content-type': 'application/json'}    
    Thread(target=writerz, args=(url, headers, jsonstring), daemon=True).start()
    # Теперь вы можете использовать GUI без блокировки основного потока
    messagebox.showinfo("Информация", "Запись данных начата.")

model_check_values = "model_check_values.pkl"
# load model from pickle file
with open(model_check_values, 'rb') as file:  
    model = pickle.load(file)

def prepare_data_for_anomalies(file_name = 'predicted_values_updated.csv'):
    df_anomalies = pd.read_csv(file_name)
    df_anomalies['DateTime'] = pd.to_datetime(df_anomalies['DateTime'])
    variables = df_anomalies.columns[1:]
    df_features = df_anomalies[variables]
    scaler_features = MinMaxScaler()
    scaler_features.fit(df_features)
    df_features = scaler_features.transform(df_features)
    df_anomalies.loc[:, variables] = df_features
    seq_size = 10
    def to_sequence(x, seq_size):
        x_values = []
        for i in range(len(x) - seq_size):
            x_values.append(x.iloc[i:(i + seq_size)].values) 
        return np.array(x_values)        
    predX = to_sequence(df_anomalies[variables], seq_size)
    max_trainMAE = 0.5
    predPredict = model.predict(predX)
    testMAE = np.mean(np.abs(predPredict - predX), axis=1)
    variables = ['ssx10mag10cp930','ssx10pab20ct001','ssx10pab10ct001','ssxtg_2ce003','ssx10lca10ct001','ssx10pab10cf001','ssx10lbg11cp920','ssx10lbg40cp920','ssx10pab60ct001','ssx10pab30ct001','ssx10maa01cp001','ssx10lca20cf001','ssx10maa01ct001','ssx10pab30cf001']
    return df_anomalies, testMAE, scaler_features, seq_size, max_trainMAE
    # Iterate over the variables and create a subplot for each one
    for i, variable in enumerate(variables):
        # Create DataFrame for the current variable
        anomaly_df = pd.DataFrame(df_anomalies[variable].iloc[seq_size:], columns=[variable])

        # Get the testMAE values for the current variable
        testMAE_values = testMAE[:, 1].tolist()

        # Add columns to the DataFrame
        anomaly_df['testMAE'] = testMAE_values
        anomaly_df['max_trainMAE'] = max_trainMAE
        anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
        anomaly_df['DateTime'] = df_anomalies[seq_size:]['DateTime']

        # Inverse transform the values for the current variable
        test_numeric = df_anomalies.drop(columns=['DateTime'])
        test_inverse = scaler_features.inverse_transform(test_numeric)
        test_inverse = test_inverse[seq_size:, variables.index(variable)].tolist()
        anomaly_df['transform_values'] = test_inverse

        # Plot the actual values and anomalies on the current subplot
        #ax = axs[i]
        sns.lineplot(x=anomaly_df['DateTime'], y=anomaly_df['transform_values'], label='transform_values', ax=ax)
        sns.scatterplot(x=anomaly_df.loc[anomaly_df['anomaly'] == True, 'DateTime'], 
                         y=anomaly_df.loc[anomaly_df['anomaly'] == True, 'transform_values'], 
                         color='r', label='anomalies', ax=ax)
        #ax.set_title(f'Anomalies for {variable}')
        #ax.set_xlabel('DateTime')
        #ax.set_ylabel(variable)
        #ax.legend()
    # Adjust the spacing between the subplots
    #plt.subplots_adjust(hspace=0.5)
    
    # Show the plot
    #plt.show()

#prepare_data_for_anomalies()




def load_and_preprocess_data(file_name):
    df = pd.read_csv(file_name)
    df.dropna(inplace=True)
    if 'DateTime' in df.columns:
        date_time = pd.to_datetime(df.pop('DateTime'), infer_datetime_format=True)
    else:
        print("Столбец 'DateTime' отсутствует в датафрейме.")
        return None, None
    
    return df, date_time

def draw_graphs(file_name= 'data_8_without_duplicates.csv'):
    try:  
     df, date_time =  load_and_preprocess_data(file_name)
     file_size = os.path.getsize(file_name)
     df.index = date_time
     selected_columns = df.columns[:2] 

     fig, axs = plt.subplots(len(selected_columns), 1, sharex=True, num="Предиктивная аналитика. Графики")

     fig.set_size_inches(20, 30)

     plt.subplots_adjust(left=0.15, bottom=0.2)
     predicted_lines = []
     lines = []
     anomaly_lines = []
     critical_points = []
     #anomaly_lines = []
     variables = ['ssx10mag10cp930','ssx10pab20ct001','ssx10pab10ct001','ssxtg_2ce003','ssx10lca10ct001','ssx10pab10cf001','ssx10lbg11cp920','ssx10lbg40cp920','ssx10pab60ct001','ssx10pab30ct001','ssx10maa01cp001','ssx10lca20cf001','ssx10maa01ct001','ssx10pab30cf001']
     for i, column in enumerate(selected_columns):
        ax = axs[i] 
        ax.figure.set_figheight(8)  # Высота каждого подграфика
        ax.figure.set_figwidth(10)  # Ширина каждого подграфика
        line, = ax.plot([], [], label=column)
        predicted_line, = ax.plot([], [], label=f'{column}_predicted', color='darkmagenta', linestyle='-')  # Линия для предсказанных значений
        anomaly_line, = ax.plot([], [], label=f'{column}_anomaly', color='darkmagenta', linestyle='-') 
        critical_point, = ax.plot([], [], 'ro', label=f'{column}_critical_points')
        lines.append(line)
        predicted_lines.append(predicted_line)
        anomaly_lines.append(anomaly_line)
        critical_points.append(critical_point)
        #ax.set_ylabel(column)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_yticks(np.arange(min(df[column])*0.5, max(df[column])*1.5, 2))
    
     for ax in axs:
        ax.grid(True)
        ax.set_xlabel('DateTime')

     def animate(i,df):
        # Получение последних данных
        last_data = df.index[i]
        data_defore_last_data = df[df.index <= last_data]

        TOTAL = df.shape[0]

        if i >= 72:
          original_df = data_defore_last_data.tail(400) if len(data_defore_last_data) >= 400 else data_defore_last_data.copy()

          data_defore_last_data_scaled, scaler = scale_data(data_defore_last_data)
          X_test = multiStepSampler(data_defore_last_data_scaled)

          predicted_values = lstm_model.predict(X_test)
          predicted_values = predicted_values[-1]
          predicted_values = scaler.inverse_transform(predicted_values)
          predicted_timestamps = pd.date_range(start=df.index[i] + pd.Timedelta(seconds=1), periods=24, freq='S')
          predicted_df = pd.DataFrame(predicted_values, columns=df.columns)
          predicted_df['DateTime'] = predicted_timestamps
          predicted_df = predicted_df.set_index('DateTime')
          predicted_df.to_csv('predicted_values_updated.csv', index=True)
          df_anomalies, testMAE, scaler_features, seq_size, max_trainMAE = prepare_data_for_anomalies()

          original_df['DateTime'] = date_time
          original_df = original_df.set_index('DateTime')
             
        print(f"Последние данные: {last_data}")
        new_file_size = os.path.getsize(file_name)
        print(new_file_size)
        
        # Обновление графика
        for j, line in enumerate(lines):
            column = selected_columns[j]
            line.set_data(df.index[:i].values, df[column][:i])
            axs[j].set_ylim(df[column].min()*0.6, df[column].max()*1.5)
            x_extend = 0.2 * (df.index[i] - df.index[0])
            axs[j].set_xlim(df.index[0] - x_extend, df.index[i] + x_extend)
            predicted_line = predicted_lines[j]
            if i > 72:
              predicted_line.set_data(predicted_df.index.values, predicted_df[column].values)
              #print(f"Линия предсказанных: {predicted_line}")
              anomaly_df = pd.DataFrame(df_anomalies[column].iloc[seq_size:], columns=[column])
              testMAE_values = testMAE[:, 1].tolist()
             # Add columns to the DataFrame
              anomaly_df['testMAE'] = testMAE_values
              anomaly_df['max_trainMAE'] = max_trainMAE
              anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
              anomaly_df['DateTime'] = df_anomalies[seq_size:]['DateTime']

        # Inverse transform the values for the current variable
              test_numeric = df_anomalies.drop(columns=['DateTime'])
              test_inverse = scaler_features.inverse_transform(test_numeric)
              test_inverse = test_inverse[seq_size:, variables.index(column)].tolist()
              anomaly_df['transform_values'] = test_inverse
              anomaly_line = anomaly_lines[j]
              #print(f"Линия проверки аномалий: {anomaly_df}")
              anomaly_line.set_data(anomaly_df['DateTime'],anomaly_df['transform_values'])
              #axs[j].scatter(
                #x=anomaly_df.loc[anomaly_df['anomaly'] == True, 'DateTime'],
                #y=anomaly_df.loc[anomaly_df['anomaly'] == True, 'transform_values'],
                #color='r', label='anomalies',
                 #)
              critical_point = critical_points[j]
              critical_point.set_data(anomaly_df.loc[anomaly_df['anomaly'] == True, 'DateTime'], anomaly_df.loc[anomaly_df['anomaly'] == True, 'transform_values'])
              print(critical_point)
        plt.pause(0.5)  

        return lines, predicted_lines, anomaly_lines, critical_points

     ani = FuncAnimation(fig, animate, frames=len(df),  fargs=(df,), interval=1, blit=False)
     plt.show()
         
    except FileNotFoundError:
        print(f"Файл '{file_name}' не найден.")


#def draw_graphs(fil
    #try:
     #df, date_time =  load_and_preprocess_data(file_name)
     #df.index = date_time
     #selected_columns = df.columns[:3] 

     #fig, axs = plt.subplots(len(selected_columns), 1, sharex=True, num="Предиктивная аналитика. Графики")

     #fig.set_size_inches(10, 20)e_name= 'data_8_without_duplicates.csv'):

     #plt.subplots_adjust(left=0.15, bottom=0.2)

     #lines = []
     #for i, column in enumerate(selected_columns):
        #ax = axs[i]
        #ax.figure.set_figheight(6)  # Высота каждого подграфика
        #ax.figure.set_figwidth(5)  # Ширина каждого подграфика
        #line, = ax.plot([], [], label=column)
        #lines.append(line)
        #ax.set_ylabel(column)
        #ax.legend()
        #ax.tick_params(axis='x', labelrotation=45)

     #for ax in axs:
        #ax.grid(True)
        #ax.set_xlabel('DateTime')

     #def animate(i):
        # Получение последних данных
        #last_data = df.index[i]
        #print(f"Последние данные: {last_data}")
        # Обновление графика
        #for j, line in enumerate(lines):
           #column = selected_columns[j]
            #line.set_data(df.index[:i+1].values, df[column][:i+1])
            #axs[j].set_ylim(df[column].min(), df[column].max())
            #axs[j].set_xlim(df.index[0], df.index[i+1])
        # Дополнительная пауза для отображения обновления
        #plt.pause(1)  
        #return lines

     #ani = FuncAnimation(fig, animate, frames=len(df), interval=1, blit=False)
     #plt.show()
     #xcept FileNotFoundError:
        #print(f"Файл '{file_name}' не найден.")



#UI окно
# Создаем главное окно
root = tk.Tk()
root.title("Предиктивная аналитика. Основное окно")
root.geometry("500x400")

# Настраиваем шрифт для кнопок
button_font = ("Times New Roman", 12)

# Добавьте кнопку для остановки записи данных


plot_button = tk.Button(root, text="Draw Grafics", command= lambda: draw_graphs('readdata10.csv'), font=button_font, width=15, height=2)
plot_button.pack(side=tk.TOP, padx=10, pady=5)

#plot_button = tk.Button(root, text="Draw Grafics", command= lambda: draw_graphs('data_8_without_duplicates.csv'), font=button_font, width=15, height=2)
#plot_button.pack(side=tk.TOP, padx=10, pady=5)

start_button = tk.Button(root, text="Start Recording", command=start_data_process, font=button_font, width=15, height=2)
start_button.pack(side=tk.TOP, padx=10, pady=5)


stop_button = tk.Button(root, text="Stop Recording", command=stop_data_process, font=button_font, width=15, height=2)
stop_button.pack(side=tk.TOP, padx=10, pady=5)


pred_button = tk.Button(root, text="Do Prediction", command=lambda: make_prediction(), font=button_font, width=15, height=2)
pred_button.pack(side=tk.TOP, padx=10, pady=5)


# Создаем рамку для разделения кнопок и графиков
frame = tk.Frame(root)
frame.pack(side=tk.TOP, padx=10, pady=10)

# Создаем два графика

#fig, axs = plt.subplots(figsize=(12, 6))  # Создаем фигуру для графика
#canvas = FigureCanvasTkAgg(fig, frame)
#canvas.get_tk_widget().pack(side=tk.LEFT, padx=10, pady=10)



lstm_model_file = "lstm_model_prediction.pkl"
# load model from pickle file
with open(lstm_model_file, 'rb') as file:  
    lstm_model = pickle.load(file)



#df = pd.read_csv('data_8_without_duplicates.csv')
import matplotlib.dates as mdates
INPUT_TIMES = 72
OUTPUT_TIMES = 24
# Single step dataset preparation

def multiStepSampler(df, window_input=INPUT_TIMES):
	xRes = []
	for i in range(0, len(df) - window_input):
		xRes.append(df.iloc[i:i + window_input].values)
	return np.array(xRes)



#X_test = multiStepSampler(df)

def scale_data(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    return pd.DataFrame(df_scaled, columns=df.columns), scaler





def plot_predictions(original_df, predicted_df, selected_columns):
    fig, axes = plt.subplots(nrows=len(selected_columns[:3]), ncols=1, figsize=(12, 4 * len(selected_columns[:3])), num="Предиктивная аналитика. Предсказания")
    fig.set_size_inches(10, 12)
    plt.subplots_adjust(left=0.15, bottom=0.2,hspace=0.2, right=0.75)
    for i, column in enumerate(selected_columns[:3]):
         
           axes[i].plot(predicted_df.index, predicted_df[column], label=column)
           axes[i].plot(original_df.index, original_df[column], label='Original ' + column)
           axes[i].set_xlabel("Время")
           axes[i].set_ylabel("Значение")
           axes[i].figure.set_figheight(6)  # Высота каждого подграфика
           axes[i].figure.set_figwidth(5)  # Ширина каждого подграфика
           axes[i].set_title(column)
           axes[i].legend()
       # Выравниваем подграфики
    plt.tight_layout()

    plt.show()


def make_prediction(file_name = 'data_8_without_duplicates.csv'):
    df, date_time = load_and_preprocess_data(file_name)
    selected_columns = df.columns[:]
    TOTAL = df.shape[0]

    if TOTAL >=72:
       original_df = df.tail(400) if len(df) >= 400 else df.copy()
       df, scaler = scale_data(df)
       test_df = df
       X_test = multiStepSampler(test_df)
       df = pd.read_csv(file_name)
       df['DateTime'] = pd.to_datetime(df['DateTime'])
       last_timestamp = pd.to_datetime(df.iloc[-1][-1])
       if 'DateTime' in df.columns:
              date_time = pd.to_datetime(df.pop('DateTime'), infer_datetime_format=True)
       else:
           print("Столбец 'DateTime' отсутствует в датафрейме.")

       predicted_values = lstm_model.predict(X_test)
       print(predicted_values.shape)
       predicted_values = predicted_values[-1]
       predicted_values = scaler.inverse_transform(predicted_values)
       predicted_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(seconds=1), periods=24, freq='S')
       predicted_df = pd.DataFrame(predicted_values, columns=selected_columns)
       print(predicted_df)
       predicted_df['DateTime'] = predicted_timestamps
       predicted_df = predicted_df.set_index('DateTime')
       # Переместить столбец DateTime в конец
       predicted_df.to_csv('predicted_values_updated.csv', index=False)       
       #predicted_df.to_csv('predicted_values.csv')
       original_df['DateTime'] = date_time
       original_df = original_df.set_index('DateTime')
       # Строим график
       plot_predictions(original_df, predicted_df, df.columns)



root.mainloop()

#def make_prediction(file_name='data_8_without_duplicates.csv'):
    #df, date_time = load_and_preprocess_data(file_name)
    #if df is None:
        #return
    #TOTAL = df.shape[0]

    #if TOTAL >= 72:
        #original_df = df.tail(400) if len(df) >= 400 else df.copy()
        #df, scaler = scale_data(df)
        #last_timestamp = pd.to_datetime(original_df.iloc[-1].name)
        #predicted_values, predicted_timestamps = create_predictions(df, scaler, last_timestamp)

        #predicted_df = pd.DataFrame(predicted_values, columns=df.columns)
        #predicted_df['DateTime'] = predicted_timestamps
        #predicted_df.set_index('DateTime', inplace=True)
        #predicted_df.to_csv('predicted_values_updated.csv')

        #riginal_df['DateTime'] = date_time
        ##plot_predictions(original_df, predicted_df, df.columns)
#predicted_df = pd.read_csv('predicted_values_updated.csv')

# Переместить столбец DateTime в конец
#predicted_df = predicted_df[['ssx10mag10cp930','ssx10pab20ct001','ssx10pab10ct001','ssxtg_2ce003','ssx10lca10ct001','ssx10pab10cf001','ssx10lbg11cp920','ssx10lbg40cp920','ssx10pab60ct001','ssx10pab30ct001','ssx10maa01cp001','ssx10lca20cf001','ssx10maa01ct001','ssx10pab30cf001','DateTime']]

# Сохранить в файл
#predicted_df.to_csv('predicted_values_updated.csv', index=False) 



