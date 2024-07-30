import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Cargar datos
try:
    # Leer el archivo CSV y manejar líneas problemáticas
    data = pd.read_csv('datos_comidas.csv', on_bad_lines='warn')
    print(data.head())
    print(data.columns)  # Verificar las columnas disponibles
except Exception as e:
    print(f"Error al leer el archivo CSV: {e}")

# Calcular IMC
data['IMC'] = data['peso'] / (data['altura'] / 100) ** 2

# Calcular calorías necesarias
def calcular_calorias(row):
    # Utilizar solo el peso, altura y edad para el cálculo de calorías
    tmb = 88.362 + (13.397 * row['peso']) + (4.799 * row['altura']) - (5.677 * row['edad'])
    
    # Suponiendo un factor de actividad moderado
    calorias = tmb * 1.55
    return calorias

data['calorias_necesarias'] = data.apply(calcular_calorias, axis=1)

# Codificar el objetivo y las comidas
le_objetivo = LabelEncoder()
data['objetivo'] = le_objetivo.fit_transform(data['objetivo'])

le_desayuno = LabelEncoder()
data['desayuno'] = le_desayuno.fit_transform(data['desayuno'])

le_almuerzo = LabelEncoder()
data['almuerzo'] = le_almuerzo.fit_transform(data['almuerzo'])

le_merienda = LabelEncoder()
data['merienda'] = le_merienda.fit_transform(data['merienda'])

le_cena = LabelEncoder()
data['cena'] = le_cena.fit_transform(data['cena'])

# Separar características y etiquetas
X = data[['peso', 'altura', 'edad', 'objetivo', 'IMC', 'calorias_necesarias']]
y_desayuno = data['desayuno']
y_almuerzo = data['almuerzo']
y_merienda = data['merienda']
y_cena = data['cena']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train_desayuno, y_test_desayuno = train_test_split(X, y_desayuno, test_size=0.2, random_state=42)
X_train, X_test, y_train_almuerzo, y_test_almuerzo = train_test_split(X, y_almuerzo, test_size=0.2, random_state=42)
X_train, X_test, y_train_merienda, y_test_merienda = train_test_split(X, y_merienda, test_size=0.2, random_state=42)
X_train, X_test, y_train_cena, y_test_cena = train_test_split(X, y_cena, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar los modelos
model_desayuno = RandomForestClassifier()
model_almuerzo = RandomForestClassifier()
model_merienda = RandomForestClassifier()
model_cena = RandomForestClassifier()

model_desayuno.fit(X_train, y_train_desayuno)
model_almuerzo.fit(X_train, y_train_almuerzo)
model_merienda.fit(X_train, y_train_merienda)
model_cena.fit(X_train, y_train_cena)

# Guardar los modelos entrenados y el scaler
joblib.dump(model_desayuno, 'model_desayuno.pkl')
joblib.dump(model_almuerzo, 'model_almuerzo.pkl')
joblib.dump(model_merienda, 'model_merienda.pkl')
joblib.dump(model_cena, 'model_cena.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Guardar los codificadores
joblib.dump(le_objetivo, 'le_objetivo.pkl')
joblib.dump(le_desayuno, 'le_desayuno.pkl')
joblib.dump(le_almuerzo, 'le_almuerzo.pkl')
joblib.dump(le_merienda, 'le_merienda.pkl')
joblib.dump(le_cena, 'le_cena.pkl')

print("Modelos entrenados y guardados exitosamente.")
