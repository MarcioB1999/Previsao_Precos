import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics as m
from tensorflow import keras
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler




class Modelo:
    def __init__(self,num):
        self.model = keras.Sequential([
                keras.layers.Dense(24,activation='relu',input_shape=[num]),
                keras.layers.Dense(24,activation='relu'),
                keras.layers.Dense(1)
                ])


    def X_transform(self,X,columnsMinMax,columnsStander):
        X_aux = pd.DataFrame({})
        scaleMinMax = []
        for column in range(len(columnsMinMax)):
            scaleMinMax.append(MinMaxScaler())
            scaleMinMax[column].fit(X[columnsMinMax[column]].values.reshape(len(X),1))
            X_aux[columnsMinMax[column]] = scaleMinMax[column].transform(X[columnsMinMax[column]].values.reshape(len(X),1)).reshape(-1)

        scaleStander = []
        for column in range(len(columnsStander)):
            scaleStander.append(StandardScaler())
            scaleStander[column].fit(X[columnsStander[column]].values.reshape(len(X),1))
            X_aux[columnsStander[column]] = scaleStander[column].transform(X[columnsStander[column]].values.reshape(len(X),1)).reshape(-1)

        return X_aux, scaleMinMax, scaleStander
    

    def Y_transform(self,Y):
        Y_aux = pd.DataFrame({})
        scale_price = StandardScaler()
        scale_price.fit(Y['price'].values.reshape(len(Y),1))
        Y_aux['price'] = scale_price.transform(Y['price'].values.reshape(len(Y),1)).reshape(-1)
        return Y_aux, scale_price


    def div_data(self,X,Y):
        X_train,X_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
        Y_train,Y_test = Y[:int(len(Y)*0.8)], Y[int(len(Y)*0.8):]
        return X_train,X_test,Y_train,Y_test


    def fit(self,X_train,Y_train):
        self.model.compile(
            optimizer = keras.optimizers.RMSprop(learning_rate=0.001),
            loss = 'mse',
            metrics = ['mae','mse','mape']
            )
        
        history = self.model.fit(
                    x=X_train,
                    y=Y_train,
                    epochs = 400,
                    verbose = 0,
                    validation_split = 0.2
                    )
        return pd.DataFrame(history.history)


    def plot_metricas(self,history):
        fig=plt.figure(figsize=(6,4))
        plt.subplots_adjust(wspace=0.3, hspace=0.6)

        ax=fig.add_subplot(3,1,1)
        ax.set_title('LOSS',fontsize=8)
        plt.plot(history.loss,label='Treino')
        plt.plot(history.val_loss,label='Validação')
        plt.legend()


        ax=fig.add_subplot(3,1,2)
        ax.set_title('MAE',fontsize=8)
        plt.plot(history.mae,label='Treino')
        plt.plot(history.val_mae,label='Validação')
        plt.legend()

        ax=fig.add_subplot(3,1,3)
        ax.set_title('MSE',fontsize=8)
        plt.plot(history.mse,label='Treino')
        plt.plot(history.val_mse,label='Validação')
        plt.legend()


    def metricas_teste(self,X_test,Y_test):
        print("metricas teste")
        metrics_test = self.model.evaluate(X_test,Y_test)
        print(f"Loss = {metrics_test[0]}")
        print(f"MAE = {metrics_test[1]}")
        print(f"MSE = {metrics_test[2]}")
        print(f"MAPE = {metrics_test[3]}")

    
    def predict(self,X):
        return self.model.predict(X).flatten()
    

    def r2(self,y,predict):
        return m.r2_score(y,predict)
    
    
    def residuos(self,y,predict):
        return y-predict
    

    def residual_plot(self,residuos,y,predict):
        fig=plt.figure(figsize=(8,6))

        plt.subplots_adjust(wspace=0.3, hspace=0.4)

        ax=fig.add_subplot(2,2,1)

        plt.scatter(y,predict)
        plt.plot([0,max(y)],[0,max(y)],color='black',lw=0.8)
        plt.xlabel('Preço', fontsize=9)
        plt.ylabel('Previsão',fontsize=9)
        plt.grid('both')



        ax1 = fig.add_subplot(2,2,2)

        sns.histplot(residuos,legend=False,kde=True,stat='proportion',label='FDP')
        plt.ylabel('FDP')
        plt.xlabel('Resíduo')
        plt.grid(axis='x')

        ax2 = ax1.twinx()
        sns.ecdfplot(residuos.reshape(-1),legend=False,ax=ax2,color='orange',label='ACF')
        plt.ylabel('FDA')
        plt.grid(axis='y')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)



        ax=fig.add_subplot(2,2,3)

        plt.plot(range(len(residuos)),residuos,marker='.', linewidth=1)
        plt. axhline(y=0,xmin=0,xmax=len(residuos),color='black', linewidth=1)
        plt.ylabel('Resíduo',fontsize=9)
        plt.xlabel('Ordem de Observação',fontsize=9)
        plt.grid('both')

        ax=fig.add_subplot(2,2,4)

        plt.scatter(predict,residuos)
        plt.axhline(y=0,xmin=0,xmax=len(predict),color='black', linewidth=1)
        plt.ylabel('Resíduo',fontsize=9)
        plt.xlabel('Previsão',fontsize=9)
        plt.grid('both')


    def predict_plot(self,y,predict):
        fig = plt.figure(figsize=(12,9))

        ax=fig.add_subplot(2,1,1)
        plt.plot(range(len(predict)),predict,label='previsao')
        plt.plot(range(len(y)),y,alpha=0.5,label='teste')
        plt.legend()
 
        ax=fig.add_subplot(2,1,2)
        sns.histplot(predict.reshape(-1),stat='proportion',label='previsão')
        sns.histplot(y.reshape(-1),alpha=0.5,color='orange',stat='proportion',label='teste')
        plt.ylabel('')
        plt.xlabel('price')
        plt.legend()