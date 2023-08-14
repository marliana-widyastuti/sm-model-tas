import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input, LSTM, Concatenate, Reshape, Lambda, Permute, LayerNormalization, Bidirectional
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
import joblib
from keras.models import load_model

def reset():
    K.clear_session()
    if int(tf.__version__.split('.')[0]) >= 2:
        tf.compat.v1.reset_default_graph()
    else:
        tf.reset_default_graph()

def ccc(y_true, x_true):
    uy, ux = K.mean(y_true), K.mean(x_true)
    sxy = tfp.stats.covariance(y_true, x_true)
    sy, sx = tfp.stats.variance(y_true), tfp.stats.variance(x_true)
    E = 2*sxy/(sy+sx+K.pow(uy-ux, 2))
    return 1-E

def crop(dimension, start, end):
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

def plot_hist(data):
    fig, ax = plt.subplots()
    ax.plot(range(len(data.history['loss'])), data.history['loss'], label='train')
    ax.plot(range(len(data.history['loss'])), data.history['val_loss'], label='validation')
    fig.legend()
    fig.show()

def mlp(inputs):
    input = Input(shape=(len(inputs)), dtype='float32')
    hidden3 = Dense(128, activation='relu', name="layer1")(input)
    hidden4 = Dense(64, activation='relu', name="layer2")(hidden3)
    hidden5 = Dense(32, activation='relu', name="layer3")(hidden4)
    hidden6 = Dense(16, activation='relu', name="layer4")(hidden5)
    mlp_out = Dense(2, activation=lambda x: relu(x, max_value=1.0))(hidden6)
    model = Model(inputs=input, outputs=mlp_out)
    return model

class model():
    def __init__(self, DFAU, DFTAS):
        self.DFAU = pd.read_csv(DFAU)  ## directory dataframe Australia dataset
        self.DFTAS = pd.read_csv(DFTAS)  ## directory dataframe Australia dataset
        self.stations = self.DFTAS['site_id'].unique().tolist()[:-1]

        smap_cols = ['smap{}'.format(str(n)) for n in range(4,8)] ## only SMAP data 4 days ago
        ssmap_cols = ['ssmap{}'.format(str(n)) for n in range(4,8)] ## only SMAP data 4 days ago
        rains_cols = ['rain']+['rain{}'.format(str(n)) for n in range(1,4)]

        self.lstm_cols = smap_cols+ssmap_cols

        self.mlp_cols = rains_cols+['tmin','tmax',
            'irri', 'past', 'agri', 'fore', 'sava',
            'elevation',
            'AWC1', 'AWC2', 'AWC3', 'AWC4', 
            'SOC1', 'SOC2', 'SOC3', 'SOC4', 
            'CLY1', 'CLY2', 'CLY3', 'CLY4']

        self.var_in = self.lstm_cols + self.mlp_cols
        self.pwd = os.getcwd()
        if self.pwd.__contains__("\\"):
            self.sep = "\\"
        else:
            self.sep = "/"
        try:
            os.mkdir(self.pwd + self.sep + 'lstm_OneAu')
        except FileExistsError:
            print("direcory already exist")
        pass
    
    def scalerAU(self):
        DF = self.DFAU
        scaler1 = MinMaxScaler()
        scaler1.fit(DF[self.var_in])
        joblib.dump(scaler1, f'scalerAU.save')
        return scaler1

    def normalizedAU(self):
        DF = self.DFAU
        DF['date']=pd.to_datetime(DF["date"]).dt.date
        DF.dropna(axis=0,inplace=True)
        scaler1 = self.scalerAU()
        DFscaled = pd.DataFrame(scaler1.transform(DF[self.var_in]))
        DFscaled = DFscaled.set_axis(self.var_in,axis=1)
        DFscaled['site_id'], DFscaled['date'], DFscaled['soilM1'], DFscaled['soilM2'] = DF['site_id'], DF['date'], DF['res030'], DF['res3060']
        return DFscaled

    def modlx(self, lr, decay, epsilon):
        reset()
        input = Input(shape=(len(self.var_in), 1), dtype='float32')
        slice1 = crop(1, 0, int(len(self.lstm_cols)))(input)
        reshaped = Reshape((2, int(len(self.lstm_cols)/2)), input_shape=(int(len(self.lstm_cols)), 1))(slice1)
        lstm_input = Permute((2, 1), input_shape=(2, int(len(self.lstm_cols)/2)))(reshaped)
        hidden1 = Bidirectional(LSTM(50, activation='relu'))(lstm_input)
        hidden1 = LayerNormalization()(hidden1)
        lstm_output = Dense(100, activation='relu')(hidden1)
        slice2 = crop(1, int(len(self.lstm_cols)), len(self.var_in))(input)
        slice2 = Reshape((len(self.var_in) - int(len(self.lstm_cols)), ),
                        input_shape=(len(self.var_in) - int(len(self.lstm_cols)), ))(slice2)
        x = Concatenate()([lstm_output, slice2])
        hidden3 = Dense(128, activation='relu')(x)
        hidden4 = Dense(64, activation='relu')(hidden3)
        hidden5 = Dense(32, activation='relu')(hidden4)
        hidden6 = Dense(16, activation='relu')(hidden5)
        mlp_out = Dense(2, activation=lambda x: relu(x, max_value=1.0))(hidden6)
        model = Model(inputs=input, outputs=mlp_out)
        model.compile(loss=ccc, optimizer=Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=decay))
        return model
    
    def one_mod(self, DFs1, lr, decay, batch_size, epsilon):
        DFtrain = DFs1[(DFs1['date']>= datetime.date(2016,1,1)) & (DFs1['date']<datetime.date(2019,1,1))] 
        DFvali = DFs1.drop(DFtrain.index)     
        tr_in = DFtrain[self.var_in].values.reshape((DFtrain[self.var_in].shape[0],
                                                DFtrain[self.var_in].shape[1],
                                                1))
        tr_out = DFtrain[['soilM1', 'soilM2']].values
        tr_in, tr_out = tr_in.astype('float32'), tr_out.astype('float32')
        va_in = DFvali[self.var_in].values.reshape((DFvali[self.var_in].shape[0],
                                            DFvali[self.var_in].shape[1],
                                            1))
        va_out = DFvali[['soilM1', 'soilM2']].values
        va_in, va_out = va_in.astype('float32'), va_out.astype('float32')
        mdl = self.modlx(lr, decay, epsilon)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        here = mdl.fit(tr_in.reshape((tr_in.shape[0], tr_in.shape[1])),
                    tr_out,
                    epochs=150,
                    verbose=2,
                    batch_size=batch_size,
                    validation_data=(va_in.reshape((va_in.shape[0], va_in.shape[1])), va_out),
                    callbacks=[callback])
        num_loss = here.history['val_loss'][-1]
        return mdl, here, DFvali, num_loss
    
    def run_OneAU(self):
        try:
            os.mkdir(self.pwd + self.sep + 'lstm_OneAu')
        except FileExistsError:
            print("direcory already exist")

        DFscaled = self.normalizedAU()
        num_loss =1
        while (num_loss >= 0.9):
            model, test, DFvali, num_loss = self.one_mod(DFscaled, 0.003, 0.8, 32, 1e-08)
            print(num_loss)

        model.save('lstm_OneAu/lstm_OneAu')
        hist = pd.DataFrame(test.history)
        DFvali.to_csv(f'{self.pwd}{self.sep}lstm_OneAu{self.sep}dfvali_lstm_OneAu.csv')
        hist.to_csv(f'{self.pwd}{self.sep}lstm_OneAu{self.sep}hist_lstm_OneAu.csv')

        plot_hist(test)
        plt.savefig('lstm_OneAu/lstm_OneAU.png')

    def scalerTAS(self):
        DF = self.DFTAS
        scaler1 = MinMaxScaler()
        scaler1.fit(DF[self.var_in])
        joblib.dump(scaler1, f'scalerTAS.save')
        return scaler1

    def normalizedTAS(self):
        DF = self.DFTAS
        DF['date']=pd.to_datetime(DF["date"]).dt.date
        DF.dropna(axis=0,inplace=True)
        scaler1 = self.scalerTAS()
        DFscaled = pd.DataFrame(scaler1.transform(DF[self.var_in]))
        DFscaled = DFscaled.set_axis(self.var_in,axis=1)
        DFscaled['site_id'], DFscaled['date'], DFscaled['soilM1'], DFscaled['soilM2'] = DF['site_id'], DF['date'], DF['res030'], DF['res3060']
        return DFscaled

    def modlx(self, lr, decay, epsilon):
        reset()
        input = Input(shape=(len(self.var_in), 1), dtype='float32')
        slice1 = crop(1, 0, int(len(self.lstm_cols)))(input)
        reshaped = Reshape((2, int(len(self.lstm_cols)/2)), input_shape=(int(len(self.lstm_cols)), 1))(slice1)
        lstm_input = Permute((2, 1), input_shape=(2, int(len(self.lstm_cols)/2)))(reshaped)
        hidden1 = Bidirectional(LSTM(50, activation='relu'))(lstm_input)
        hidden1 = LayerNormalization()(hidden1)
        lstm_output = Dense(100, activation='relu')(hidden1)
        slice2 = crop(1, int(len(self.lstm_cols)), len(self.var_in))(input)
        slice2 = Reshape((len(self.var_in) - int(len(self.lstm_cols)), ),
                        input_shape=(len(self.var_in) - int(len(self.lstm_cols)), ))(slice2)
        x = Concatenate()([lstm_output, slice2])
        hidden3 = Dense(128, activation='relu')(x)
        hidden4 = Dense(64, activation='relu')(hidden3)
        hidden5 = Dense(32, activation='relu')(hidden4)
        hidden6 = Dense(16, activation='relu')(hidden5)
        mlp_out = Dense(2, activation=lambda x: relu(x, max_value=1.0))(hidden6)
        model = Model(inputs=input, outputs=mlp_out)
        model.compile(loss=ccc, optimizer=Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=decay))
        return model

    def LOOCV(self, probe, lr, decay, batch_size, epsilon):
        DFs1 = self.normalizedTAS()
        stations_rem = [n for n in self.stations if n != probe]
        vali_ix = np.random.randint(0, len(stations_rem))
        vali_station = self.stations[vali_ix]
        DFtrain = DFs1[(DFs1['site_id'] != probe) & (DFs1['site_id'] != vali_station)]
        DFvali = DFs1[DFs1['site_id'] == vali_station] # for early stop and fine tuning
        DFtest = DFs1[DFs1['site_id'] == probe] # for performance
        tr_in = DFtrain[self.var_in].values.reshape((DFtrain[self.var_in].shape[0],
                                                DFtrain[self.var_in].shape[1],
                                                1))
        tr_out = DFtrain[['soilM1', 'soilM2']].values
        tr_in, tr_out = tr_in.astype('float32'), tr_out.astype('float32')
        va_in = DFvali[self.var_in].values.reshape((DFvali[self.var_in].shape[0],
                                            DFvali[self.var_in].shape[1],
                                            1))
        va_out = DFvali[['soilM1', 'soilM2']].values
        va_in, va_out = va_in.astype('float32'), va_out.astype('float32')
        mdl = self.modlx(lr, decay, epsilon)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        here = mdl.fit(tr_in.reshape((tr_in.shape[0], tr_in.shape[1])),
                    tr_out,
                    epochs=150,
                    verbose=0,
                    batch_size=batch_size,
                    validation_data=(va_in.reshape((va_in.shape[0], va_in.shape[1])), va_out),
                    callbacks=[callback])
        num_loss = here.history['val_loss'][-1]
        return mdl, here, DFvali, DFtest, num_loss
    
    def add_predictions(self, df, model):
        predictions = model.predict(df[self.var_in].values)
        df['pred1'], df['pred2'] = predictions[:, 0], predictions[:, 1]
        return df
    
    def run_lstmTAS(self):
        try:
            os.mkdir(self.pwd + self.sep + 'lstmTAS')
        except FileExistsError:
            print("direcory already exist")
        for i, sta in enumerate(self.stations):
            num_loss =1
            while (num_loss >= 0.9):
                model, test, DFvali, DFtest, num_loss = self.LOOCV(sta, 0.0001, 0.8, 128, 1e-08)
                print(num_loss)

            model.save(f'lstmTAS{self.sep}lstmTAS_{sta}')
            hist = pd.DataFrame(test.history)
            DFtest2 = self.add_predictions(DFtest, model)
            
            DFvali.to_csv(f'{self.pwd}{self.sep}lstmTAS{self.sep}DFvali_lstmTAS_{sta}.csv')
            hist.to_csv(f'{self.pwd}{self.sep}lstmTAS{self.sep}hist_lstmTAS_{sta}.csv')
            DFtest2.to_csv(f'{self.pwd}{self.sep}lstmTAS{self.sep}DFtest_lstmTAS_{sta}.csv')
            
            # plot_hist(test)
            # plt.show()
            print(str(i))

    def LOOCV_TL(self, probe, lr, batch_size):
        DFs1 = self.normalizedTAS()
        stations_rem = [n for n in self.stations if n != probe]
        vali_ix = np.random.randint(0, len(stations_rem))
        vali_station = self.stations[vali_ix]
        DFtrain = DFs1[(DFs1['site_id'] != probe) & (DFs1['site_id'] != vali_station)]
        DFvali = DFs1[DFs1['site_id'] == vali_station] # for early stop and fine tuning
        DFtest = DFs1[DFs1['site_id'] == probe] # for performance
        tr_in = DFtrain[self.var_in].values.reshape((DFtrain[self.var_in].shape[0],
                                                DFtrain[self.var_in].shape[1],
                                                1))
        tr_out = DFtrain[['soilM1', 'soilM2']].values
        tr_in, tr_out = tr_in.astype('float32'), tr_out.astype('float32')
        va_in = DFvali[self.var_in].values.reshape((DFvali[self.var_in].shape[0],
                                            DFvali[self.var_in].shape[1],
                                            1))
        va_out = DFvali[['soilM1', 'soilM2']].values
        va_in, va_out = va_in.astype('float32'), va_out.astype('float32')
        
        reset()
        pretrained = load_model(self.pwd+'/lstm_OneAu/lstm_OneAu/', custom_objects={'ccc':ccc})

        for l in pretrained.layers[10:13]:
            l.trainable = False
        pretrained.compile(loss=ccc, optimizer=Adam(learning_rate=lr))
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        here = pretrained.fit(tr_in.reshape((tr_in.shape[0], tr_in.shape[1])),
                    tr_out,
                    epochs=150,
                    verbose=0,
                    batch_size=batch_size,
                    validation_data=(va_in.reshape((va_in.shape[0], va_in.shape[1])), va_out),
                    callbacks=[callback])
        num_loss = here.history['val_loss'][-1]
        return pretrained, here, DFvali, DFtest, num_loss
            
    def run_lstmTL(self):
        try:
            os.mkdir(self.pwd + self.sep + 'lstmTL')
        except FileExistsError:
            print("direcory already exist")

        for i, sta in enumerate(self.stations):
            num_loss =1
            while (num_loss >= 0.9):
                model, test, DFvali, DFtest, num_loss = self.LOOCV_TL(sta, 0.0001, 128)
                print(num_loss)
            
            model.save(f'lstmTL{self.sep}lstmTL_{sta}')
            hist = pd.DataFrame(test.history)
            DFtest2 = self.add_predictions(DFtest, model)
            
            DFvali.to_csv(f'{self.pwd}{self.sep}lstmTL{self.sep}DFvali_lstmTL_{sta}.csv')
            hist.to_csv(f'{self.pwd}{self.sep}lstmTL{self.sep}hist_lstmTL_{sta}.csv')
            DFtest2.to_csv(f'{self.pwd}{self.sep}lstmTL{self.sep}dftest_lstmTL_{sta}.csv')
            
            # plot_hist(test)
            # plt.show()
            print(str(i))
    
    def mlpOneAU(self):
        DFs1 = self.normalizedAU()
        DFtrain = DFs1[(DFs1['date']>= datetime.date(2016,1,1)) & (DFs1['date']<datetime.date(2019,1,1))] 
        DFvali = DFs1.drop(DFtrain.index)     
        tr_in = DFtrain[self.var_in].values.reshape((DFtrain[self.var_in].shape[0],
                                                DFtrain[self.var_in].shape[1],
                                                1))
        tr_out = DFtrain[['soilM1', 'soilM2']].values
        tr_in, tr_out = tr_in.astype('float32'), tr_out.astype('float32')
        va_in = DFvali[self.var_in].values.reshape((DFvali[self.var_in].shape[0],
                                            DFvali[self.var_in].shape[1],
                                            1))
        va_out = DFvali[['soilM1', 'soilM2']].values
        va_in, va_out = va_in.astype('float32'), va_out.astype('float32')
        mdl = mlp(self.var_in)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        mdl.compile(loss=ccc, optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999))
        here = mdl.fit(tr_in.reshape((tr_in.shape[0], tr_in.shape[1])),
                    tr_out,
                    epochs=500,
                    verbose=1,
                    batch_size=256,
                    validation_data=(va_in.reshape((va_in.shape[0], va_in.shape[1])), va_out),
                    callbacks=[callback])
        num_loss = here.history['val_loss'][-1]
        return mdl, here, DFvali, num_loss
    def run_mlpOneAU(self):
        try:
            os.mkdir(self.pwd + self.sep + 'mlp_OneAu')
        except FileExistsError:
            print("direcory already exist")

        num_loss =1
        while (num_loss >= 0.9):
            model, test, DFvali, num_loss = self.mlpOneAU()
            print(num_loss)

        model.save('mlp_OneAu/mlp_OneAu')
        hist = pd.DataFrame(test.history)

        DFvali.to_csv(f'{self.pwd}{self.sep}mlp_OneAu{self.sep}dfvali_mlp_OneAu.csv')
        hist.to_csv(f'{self.pwd}{self.sep}mlp_OneAu{self.sep}hist_mlp_OneAu.csv')

        # plot_hist(test)
        # plt.show()

    def mlpTAS(self, probe, lr, batch_size):
        DFs1 = self.normalizedTAS()
        stations_rem = [n for n in self.stations if n != probe]
        vali_ix = np.random.randint(0, len(stations_rem))
        vali_station = self.stations[vali_ix]
        DFtrain = DFs1[(DFs1['site_id'] != probe) & (DFs1['site_id'] != vali_station)]
        DFvali = DFs1[DFs1['site_id'] == vali_station] # for early stop and fine tuning
        DFtest = DFs1[DFs1['site_id'] == probe] # for performance
        tr_in = DFtrain[self.var_in].values.reshape((DFtrain[self.var_in].shape[0],
                                                DFtrain[self.var_in].shape[1],
                                                1))
        tr_out = DFtrain[['soilM1', 'soilM2']].values
        tr_in, tr_out = tr_in.astype('float32'), tr_out.astype('float32')
        va_in = DFvali[self.var_in].values.reshape((DFvali[self.var_in].shape[0],
                                            DFvali[self.var_in].shape[1],
                                            1))
        va_out = DFvali[['soilM1', 'soilM2']].values
        va_in, va_out = va_in.astype('float32'), va_out.astype('float32')
        mdl = mlp(self.var_in)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        mdl.compile(loss=ccc, optimizer=Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999))
        here = mdl.fit(tr_in.reshape((tr_in.shape[0], tr_in.shape[1])),
                    tr_out,
                    epochs=150,
                    verbose=0,
                    batch_size=batch_size,
                    validation_data=(va_in.reshape((va_in.shape[0], va_in.shape[1])), va_out),
                    callbacks=[callback])
        num_loss = here.history['val_loss'][-1]
        return mdl, here, DFvali, DFtest, num_loss

    def run_mlpTAS(self):
        try:
            os.mkdir(self.pwd + self.sep + 'mlpTAS')
        except FileExistsError:
            print("direcory already exist")

        for i, sta in enumerate(self.stations):
            num_loss =1
            while (num_loss >= 0.9):
                model, test, DFvali, DFtest, num_loss = self.mlpTAS(sta, 0.0001, 128)
            
            model.save(f'mlpTAS{self.sep}mlpTAS_{sta}')
            hist = pd.DataFrame(test.history)
            DFtest2 = self.add_predictions(DFtest, model)
            
            DFvali.to_csv(f'{self.pwd}{self.sep}mlpTAS{self.sep}dfvali_mlpTAS_{sta}.csv')
            hist.to_csv(f'{self.pwd}{self.sep}mlpTAS{self.sep}hist_mlpTAS_{sta}.csv')
            DFtest2.to_csv(f'{self.pwd}{self.sep}mlpTAS{self.sep}dftest_mlpTAS_{sta}.csv')
            
            # plot_hist(test)
            # plt.show()
            print(str(i))
                
    def mlpTL(self, probe, lr, batch_size):
        DFs1 = self.normalizedTAS()
        stations_rem = [n for n in self.stations if n != probe]
        vali_ix = np.random.randint(0, len(stations_rem))
        vali_station = self.stations[vali_ix]
        DFtrain = DFs1[(DFs1['site_id'] != probe) & (DFs1['site_id'] != vali_station)]
        DFvali = DFs1[DFs1['site_id'] == vali_station] # for early stop and fine tuning
        DFtest = DFs1[DFs1['site_id'] == probe] # for performance
        tr_in = DFtrain[self.var_in].values.reshape((DFtrain[self.var_in].shape[0],
                                                DFtrain[self.var_in].shape[1],
                                                1))
        tr_out = DFtrain[['soilM1', 'soilM2']].values
        tr_in, tr_out = tr_in.astype('float32'), tr_out.astype('float32')
        va_in = DFvali[self.var_in].values.reshape((DFvali[self.var_in].shape[0],
                                            DFvali[self.var_in].shape[1],
                                            1))
        va_out = DFvali[['soilM1', 'soilM2']].values
        va_in, va_out = va_in.astype('float32'), va_out.astype('float32')
        
        reset()
        pretrained = load_model(self.pwd+'/mlp_OneAu/mlp_OneAu/', custom_objects={'ccc':ccc})

        for l in pretrained.layers[:3]:
            l.trainable = False
        pretrained.compile(loss=ccc, optimizer=Adam(learning_rate=lr))
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        here = pretrained.fit(tr_in.reshape((tr_in.shape[0], tr_in.shape[1])),
                    tr_out,
                    epochs=150,
                    verbose=0,
                    batch_size=batch_size,
                    validation_data=(va_in.reshape((va_in.shape[0], va_in.shape[1])), va_out),
                    callbacks=[callback])
        num_loss = here.history['val_loss'][-1]
        return pretrained, here, DFvali, DFtest, num_loss

    def run_mlpTL(self):
        try:
            os.mkdir(self.pwd + self.sep + 'mlpTL')
        except FileExistsError:
            print("direcory already exist")

        for i, sta in enumerate(self.stations):
            num_loss =1
            while (num_loss >= 0.9):
                model, test, DFvali, DFtest, num_loss = self.mlpTL(sta, 0.0001, 128)
            
            model.save(f'mlpTL{self.sep}mlpTL_{sta}')
            hist = pd.DataFrame(test.history)
            DFtest2 = self.add_predictions(DFtest, model)
            
            DFvali.to_csv(f'{self.pwd}{self.sep}mlpTL{self.sep}dfvali_mlpTL_{sta}.csv')
            hist.to_csv(f'{self.pwd}{self.sep}mlpTL{self.sep}hist_mlpTL_{sta}.csv')
            DFtest2.to_csv(f'{self.pwd}{self.sep}mlpTL{self.sep}dftest_mlpTL_{sta}.csv')
            
            # plot_hist(test)
            # plt.show()
            print(str(i))                   

def calculate_execution_time(start: float, stop: float):
    if stop - start < 60:
        execution_duration = ("%1d" % (stop - start))
        print(f"Process completed in {execution_duration} seconds")
        exit(0)
    elif stop - start < 3600:
        execution_duration = ("%1d" % ((stop - start) / 60))
        print(f"Process completed in {execution_duration} minutes")
        exit(0)
    else:
        execution_duration = ("%1d" % ((stop - start) / 3600))
        print(f"Process complete in {execution_duration} hours")
        exit(0)

