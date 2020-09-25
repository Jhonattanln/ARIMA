import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

## Importando os dados
df = pd.read_excel(r'C:\Users\Jhona\OneDrive\Documentos\Projetos\ARIMA\ARIMA\ARIMA\economatica.xlsx', index_col=0)
df.rename(columns={'CPLR6':'CPLE6'}, inplace=True)

## Ajustando os dados
klbn = df['KLBN11'].loc['4T2005':].astype('float')
klbn1 = pd.DataFrame(klbn)

## Transformando em uma série estacionária
shift = klbn1.shift(1)/klbn1
shift = shift.dropna()

model = pm.auto_arima(klbn1, seasonal=False,
                      d=2, trend='c', max_p=5,
                      max_q=5, trace=True,
                      error_action='ignore',
                      suppress_warnings=True)

print(model.summary())

## Fazendos as previsões futuras
arima = SARIMAX(klbn1, order=(2, 2, 2))
arima_results = arima.fit()
arima_forecast = arima_results.get_forecast(steps=1).predicted_mean
print(arima_forecast)

## PLotando o gráfico de previsão
plt.style.use('ggplot')
plt.plot(klbn1.index, klbn1, label='Divulgado')
plt.plot(arima_forecast.index, arima_forecast, label='Previsão', linestyle='--', color='b')
plt.xticks(rotation=45)
plt.show()