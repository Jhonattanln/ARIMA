import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller

### Importando dados
df = pd.read_excel(r'economatica.xlsx', index_col=0)
df.rename(columns={'CPLR6':'CPLE6'}, inplace=True)

#### Ajustando e plotando a série
klbn = df['KLBN11'].loc['4T2005':].astype('float')
klbn1 = pd.DataFrame(klbn)
plt.style.use('ggplot')
plt.plot(klbn1)
plt.xticks(rotation=45)
plt.show()

### Teste ADF
teste = adfuller(shift)
print(teste)

##Fazendo a diff
klbn_diff = klbn1.diff().dropna()
print(klbn_diff)
klbn_diff2 = klbn.diff().diff().dropna()
print(klbn_diff)
plt.plot(klbn_diff2)
plt.xticks(rotation=45)
plt.show()

## Testando com outros metodos
shift = klbn1.shift(1)/klbn1
shift = shift.dropna()
plt.plot(shift)
plt.show()


## Teste ADF para a série diff
teste_diff = adfuller(klbn_diff)
print(teste_diff)

teste_diff2 = adfuller(klbn_diff2)
print(teste_diff2)

## Gráficos ACF e PACF
fig, (ax1, ax2) = plt.subplots(2, 1)
plot_acf(klbn_diff2, lags=15, zero=False, ax=ax1)
plot_pacf(klbn_diff2, lags=15, zero=False, ax=ax2)
plt.show()

## Estimando o modelo
model = SARIMAX(klbn, order=(1, 2, 2))
result = model.fit()
print(result.summary())
forecast = result.get_prediction(start=-10)
mean_forecast = forecast.predicted_mean
print(mean_forecast)

## Gráficos do modelo
one_step_forecast = result.get_prediction(start=-20)
mean_forecast = one_step_forecast.predicted_mean
confidence_intervals = one_step_forecast.conf_int()
lower_limits = confidence_intervals.loc[:, 'lower KLBN11']
upper_limits = confidence_intervals.loc[:,'upper KLBN11']
plt.plot(klbn.index, klbn, label='Observado')
plt.plot(mean_forecast.index, mean_forecast, color='b', linestyle='--', label='Previsão')
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color='pink')
plt.xticks(rotation = 45)
plt.show()

## Plotando as estatisticas
result.plot_diagnostics()
plt.show()

## Previsões dinâmicas
dynamic_forecast = result.get_prediction(start=-5, synamic=True)
mean_forecast1 = dynamic_forecast.predicted_mean
confidence_intervals1 = dynamic_forecast.conf_int()

lower_limits1 = confidence_intervals1.loc[:,'lower KLBN11']
upper_limits1 = confidence_intervals1.loc[:, 'upper KLBN11']
print(mean_forecast1)