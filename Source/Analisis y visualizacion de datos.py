#!/usr/bin/env python
# coding: utf-8

# In[194]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose


# In[195]:


df = pd.read_csv('sales.csv')
df


# In[196]:


df.info()


# In[197]:


df.describe()


# In[198]:


valores_perdidos = df.isnull().sum()
valores_perdidos


# In[199]:


df[df.isnull().any(axis=1)]


# In[200]:


df_limpio = df.dropna(subset=['Order ID', 'Product', 'Quantity Ordered', 'Price Each', 'Order Date', 'Purchase Address'])


# In[201]:


df_limpio.describe()


# In[202]:


print(df['Price Each'].unique())
print("---------------------------")
print(df['Quantity Ordered'].unique())
print("---------------------------")


# In[203]:


import pandas as pd

# Filtrar filas que contienen 'Price Each' o 'Quantity Ordered'
df = df[~df["Price Each"].isin(["Price Each"])]
df = df[~df["Quantity Ordered"].isin(["Quantity Ordered"])]
df = df[~df["Purchase Address"].isin(["Purchase Address"])]


# Eliminar NaNs en las columnas relevantes
df = df.dropna(subset=['Order ID', 'Product', 'Quantity Ordered', 'Price Each', 'Order Date', 'Purchase Address'])

# Convertir a valores numéricos
df["Price Each"] = pd.to_numeric(df["Price Each"])
df["Quantity Ordered"] = pd.to_numeric(df["Quantity Ordered"])

# Verificar los valores únicos después de la limpieza
print(df["Price Each"].unique())
print("---------------------------")
print(df["Quantity Ordered"].unique())
print("---------------------------")


# In[204]:


### Analisis descriptivo


# In[205]:


df.describe()


# In[206]:


conteo_producto = df['Product'].value_counts()
conteo_producto


# In[207]:


df['City'] = df['Purchase Address'].apply(lambda x: x.split(',')[1].strip())
df['Order Date'] = df['Order Date'].apply(lambda x: x.split(' ')[0].strip())
df


# In[208]:


conteo_ciudades = df['City'].value_counts().sort_index()
conteo_ciudades


# In[209]:


df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df


# In[210]:


conteo_fechas = df['Order Date'].dt.month.value_counts().sort_index()
conteo_fechas


# In[211]:


### visualizar datos descriptivos


# In[212]:


#Distribucion de precios
plt.figure(figsize=(8,6))
sns.histplot(df['Price Each'], bins=15, kde=True)
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
plt.title('Distribucion de precios')
plt.show()


# In[213]:


#Conteo de productos vendidos
plt.figure(figsize=(8,6))
sns.countplot(data = df, y=df['Product'], order = df['Product'].value_counts().index)
plt.xlabel('Cantidad vendida')
plt.ylabel('Producto')
plt.title('Cantidad vendida por producto')
plt.show()


# In[214]:


# Boxplot de precios
plt.figure(figsize=(8,6))
sns.boxplot(x = df['Price Each'])
plt.xlabel('Precio unitario')
plt.title('Boxplot de precios')
plt.show()


# In[215]:


# ventas por ciudad
city_order = df['City'].value_counts().index
plt.figure(figsize=(8,6))
sns.countplot(data = df, y='City', order = city_order)
plt.xlabel('Cantidad vendidas')
plt.ylabel('Ciudad')
plt.title('Cantidad vendida por ciudad')
plt.show()


# In[216]:


#ventas mensuales por mes
df['Month'] = df['Order Date'].dt.month
ventas_mensuales = df['Month'].value_counts().sort_index()
plt.figure(figsize=(8,6))
plt.plot(ventas_mensuales.index, ventas_mensuales.values, marker='o')
plt.xlabel('Mes')
plt.ylabel('Cantidad vendida')
plt.title('Cantidad vendida por mes')
plt.show()


# ### ANALISIS DE FRECUENCIAS Y PROPORCIONES 

# In[217]:


frecuencia_compra = df['Product'].value_counts()
frecuencia_ciudades = df['City'].value_counts()
proporcion_compra = df['Product'].value_counts(normalize=True)
proporcion_ciudades = df['City'].value_counts(normalize=True)
print(frecuencia_compra)
print(frecuencia_ciudades)
print(proporcion_compra)
print(proporcion_ciudades)


# In[218]:


# tablas cruzada (unidades de productos vendidas por ciudad)
tabla_cruzada = pd.crosstab(df['City'], df['Product'])
tabla_cruzada


# In[219]:


#esto es lo mismo que lo de arriba
producto_ciudad = df.groupby('City')['Product'].value_counts()
producto_ciudad


# In[220]:


productos_caros = df.loc[df['Price Each'] > 1000]
productos_caros['City'].value_counts(normalize=True)


# ### GRAFICOS 

# In[221]:


#grafico de barras de unidades de producto vendidas  
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Product', order=df['Product'].value_counts().index)
plt.xlabel('Productos')
plt.ylabel('Unidades vendidas')
plt.title('Unidades vendidas de cada producto')
plt.xticks(rotation=90)
plt.grid(True, alpha=0.6)
plt.show()


# In[222]:


#grafico de barras de unidades de producto vendidas
#igual que el de arriba pero cambiando la x por la y
plt.figure(figsize=(6,4))
sns.countplot(data=df, y='Product', order=df['Product'].value_counts().index)
plt.xlabel('Productos')
plt.ylabel('Unidades vendidas')
plt.title('Unidades vendidas de cada producto')
plt.grid(True, alpha=0.6)
plt.show()


# In[223]:


# frecuecnia de producos vendidos por ciudad
# grafico de barras de unidades de producto vendidas  
plt.figure(figsize=(10,8))
sns.countplot(data=df, x='City', hue ='Product')
plt.xlabel('Ciudades')
plt.ylabel('Unidades vendidas')
plt.title('Unidades de Producto vendidas en cada ciudad')
plt.xticks(rotation=45)
plt.legend(title='Productos', loc='upper left')
plt.grid(True, alpha=0.6)
plt.show()


# In[224]:


# grafico de barras de las proporciones de productos vendidos

proporcion_producto = df['Product'].value_counts(normalize=True).reset_index()
proporcion_producto.columns = ['Product', 'Proporcion']

plt.figure(figsize=(10,8))
sns.barplot(data=proporcion_producto, y='Product', x ='Proporcion', palette = 'viridis')
plt.xlabel('Productos')
plt.ylabel('Proporcion de unidades vendidas')
plt.title('Proporciones de unidades de Producto vendidas')
plt.grid(True, alpha=0.6)
plt.show()


# In[225]:


get_ipython().run_line_magic('pinfo', 'plt.pie')


# In[226]:


# diagramas de Tarta de productos vendidos (con porcentajes)
productos = df['Product'].value_counts()
color = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
explode = [0.1]+[0] * (len(productos)-1)
plt.figure(figsize=(10,8))
plt.pie(productos, labels = productos.index, autopct = '%1.1f%%', startangle = 90, rotatelabels = True, colors = color, explode=explode, shadow=True)
plt.axis('equal')
plt.show()


# In[227]:


# diagramas de Tarta de la distribucion de las ventas por ciudad(con porcentajes)
ciudad = df['City'].value_counts()
color = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
explode = [0.1]+[0] * (len(productos)-1)
plt.figure(figsize=(10,8))
plt.pie(ciudad, labels = ciudad.index, autopct = '%1.1f%%', startangle = 90, rotatelabels = True)
plt.title('Distribucion de ventas por ciudad')
plt.axis('equal')
plt.show()


# In[228]:


# diagramas de Tarta de la distribucion de las ventas por ciudad (las que superen el limite)
limite = 15000
ciudad['Otros'] = ciudad[ciudad < limite].sum()
ciudad = ciudad[ciudad >= limite]

plt.figure(figsize=(10,8))
plt.pie(ciudad, labels = ciudad.index, autopct = '%1.1f%%', startangle = 90, rotatelabels = True)
plt.title('Distribucion de ventas por ciudad')
plt.axis('equal')
plt.show()


# ### Analisis de correlacion 

# In[229]:


correlacion = df[['Order ID', 'Quantity Ordered', 'Price Each']].corr()
plt.figure(figsize=(8,4))
sns.heatmap(correlacion, annot=True, cmap = 'coolwarm', linewidth = 0.5)
plt.title('Correlacion entre las variables')
plt.show()


# In[230]:


valor_correlacion = df['Price Each'].corr(df['Quantity Ordered'])
valor_correlacion


# ### graficos dispersion 

# In[231]:


#dispersion del numero de productos vendidos en funcion de su precio
plt.figure(figsize=(10,8))
plt.scatter(df['Price Each'], df['Quantity Ordered'], alpha=0.5)
plt.xlabel('Precio')
plt.ylabel('Cantidad de unidades vendidas')
plt.title('Unidades de Producto vendidas por precio')
plt.grid(True, alpha=0.6)
plt.show()


# In[232]:


#dispersion de los productos vendidos en funcion de su precio (con linea de tendencia)

plt.figure(figsize=(10,8))
sns.scatterplot(x='Price Each', y='Quantity Ordered', hue='Product', data=df)
sns.regplot(data=df, x='Price Each', y = 'Quantity Ordered', scatter_kws={'alpha':0.5}, color='red')
plt.xlabel('Precio')
plt.ylabel('Cantidad de unidades vendidas')
plt.title('Unidades de Producto vendidas por precio')
plt.legend(bbox_to_anchor =(1.05,1), loc='upper left')
plt.grid(True, alpha=0.6)
plt.show()


# In[233]:


#dispersion de los cantidades vendidas en funcion de su precio (con s=100 es para el tamaño)

plt.figure(figsize=(10,8))
sns.scatterplot(x='Price Each', y='Quantity Ordered', s=100, alpha=0.6, edgecolor = 'red', data=df)
plt.xlabel('Precio')
plt.ylabel('Cantidad de unidades vendidas')
plt.title('Unidades de Producto vendidas por precio')
plt.grid(True, alpha=0.6, linestyle = '--')
plt.show()


# ### Regresion lineal y multiple

# In[234]:


#regresion lineal
X = df[['Price Each']] #variable independiente
y = df['Quantity Ordered'] #variable dependiente

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.figure(figsize=(10,8))
sns.scatterplot(x='Price Each', y='Quantity Ordered', data=df)
plt.plot(X, y_pred, color='red', linewidth=2)
plt.xlabel('Precio')
plt.ylabel('Cantidad de unidades vendidas')
plt.title('Regresion lineal entre el precio y la cantidad pedida')
plt.grid(True, alpha=0.6)
plt.show()

#la linea nos enseña como seria una relacion lineal entre las dos variables


# In[235]:


#evaluar el modelo (este modelo solamente es capaz de explicar el 2,1% de los datos)
r_squared = model.score(X,y)
r_squared


# In[236]:


#Regresion multiple

df['Order_month'] = pd.to_datetime(df['Order Date'], errors = 'coerce').dt.month

X = df[['Price Each', 'Order_month']] #variable independiente
y = df['Quantity Ordered'] #variable dependiente

model_multi = LinearRegression()
model_multi.fit(X, y)
y_pred = model_multi.predict(X)


r_squared = model_multi.score(X, y)
print(r_squared)
print(model_multi.intercept_)
print(model_multi.coef_)

#este modelo solo explica el 2,1% de los datos


# In[237]:


#Calculamos los residuos (para verificar que los datos estan distribuidos aleatoriamente, lo que indica que el modelo es adecuado)
residuos = y-model_multi.predict(X)

plt.figure(figsize=(10,8))
sns.scatterplot(x=model_multi.predict(X), y=residuos)
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Grafico de Residuos')
plt.grid(True, alpha=0.6)
plt.axhline(0, linestyle = '--', color='red')
plt.show()


# In[238]:


## deberian de distribuirse aleatoriamente cerca de la linea roja. no es asi, supongo que el modelo es insuficiente


# ### Analisis de datos temporales

# In[239]:


df['Order Date'] = pd.to_datetime(df['Order Date'], errors= 'coerce')
df.info()


# In[240]:


df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day
df['Day of week'] = df['Order Date'].dt.day_name()
df['Hour'] = df['Order Date'].dt.hour
df.set_index('Order Date', inplace=True)
df


# In[241]:


#grafico de las ventas por mes

ventas_mes = df['Quantity Ordered'].resample('M').sum()
print(ventas_mes)
plt.figure(figsize=(6,4))
ventas_mes.plot()
plt.xlabel('Fechas')
plt.ylabel('Cantidad de ventas')
plt.title('Ventas mensuales')
plt.show()


# In[242]:


#ventas por mes con Seaborn

plt.figure(figsize=(6,4))
sns.lineplot(data=ventas_mes)
plt.xlabel('Fechas')
plt.ylabel('Cantidad de ventas')
plt.title('Ventas mensuales')
plt.show()


# ### Descomponer series temporales

# In[243]:


# se necesitan 2 ciclos minimo. por lo que si tenemos datos de 12 meses, tenemos que descomponerlo como minimo en 6 meses
descomponer = seasonal_decompose(ventas_mes, model='additive', period=4)
descomponer.plot()


# ### comparar ventas por productos en concreto

# In[245]:


plt.figure(figsize=(6,4))
df[df['Product'] == 'Macbook Pro Laptop']['Quantity Ordered'].resample('M').sum().plot(label='Macbook')
df[df['Product'] == 'Google Phone']['Quantity Ordered'].resample('M').sum().plot(label='Google Phone')
plt.xlabel('Fechas')
plt.ylabel('Cantidad de ventas')
plt.title('Comparacion de ventas mensuales entre Macbook y Google Phone')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




