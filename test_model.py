import joblib
import numpy as np

# Cargar los modelos, el scaler y los codificadores
model_desayuno = joblib.load('model_desayuno.pkl')
model_almuerzo = joblib.load('model_almuerzo.pkl')
model_merienda = joblib.load('model_merienda.pkl')
model_cena = joblib.load('model_cena.pkl')
scaler = joblib.load('scaler.pkl')

le_objetivo = joblib.load('le_objetivo.pkl')
le_desayuno = joblib.load('le_desayuno.pkl')
le_almuerzo = joblib.load('le_almuerzo.pkl')
le_merienda = joblib.load('le_merienda.pkl')
le_cena = joblib.load('le_cena.pkl')

# Datos de prueba
peso = 80
altura = 180
edad = 23
objetivo = 'ganar_masa'
sexo = 'hombre'  # Añadir esta característica si es necesaria

# Codificar el objetivo
objetivo_codificado = le_objetivo.transform([objetivo])[0]

# Crear un array de entrada que incluya todas las características requeridas
# Asegúrate de que el número de características coincida con el número esperado por el scaler
X = np.array([[peso, altura, edad, objetivo_codificado, 0, 0]])  # Añadir valores para las características faltantes
X_scaled = scaler.transform(X)

# Hacer las predicciones
prediccion_desayuno = model_desayuno.predict(X_scaled)
prediccion_almuerzo = model_almuerzo.predict(X_scaled)
prediccion_merienda = model_merienda.predict(X_scaled)
prediccion_cena = model_cena.predict(X_scaled)

# Decodificar las predicciones
desayuno_decodificado = le_desayuno.inverse_transform(prediccion_desayuno)
almuerzo_decodificado = le_almuerzo.inverse_transform(prediccion_almuerzo)
merienda_decodificado = le_merienda.inverse_transform(prediccion_merienda)
cena_decodificado = le_cena.inverse_transform(prediccion_cena)

# Calcular IMC
imc = peso / (altura / 100) ** 2

# Función para calcular calorías necesarias
def calcular_calorias(peso, altura, edad, objetivo):
    if objetivo == 'perder_peso':
        return (10 * peso) + (6.25 * altura) - (5 * edad) - 161
    elif objetivo == 'mantener_peso':
        return (10 * peso) + (6.25 * altura) - (5 * edad) + 5
    elif objetivo == 'ganar_masa':
        return (10 * peso) + (6.25 * altura) - (5 * edad) + 500
    else:
        return (10 * peso) + (6.25 * altura) - (5 * edad) + 200

calorias_necesarias = calcular_calorias(peso, altura, edad, objetivo)

# Mostrar resultados
print(f'Desayuno: {desayuno_decodificado[0]}')
print(f'Almuerzo: {almuerzo_decodificado[0]}')
print(f'Merienda: {merienda_decodificado[0]}')
print(f'Cena: {cena_decodificado[0]}')
print(f'IMC: {imc:.2f}')
print(f'Calorías necesarias por día: {calorias_necesarias:.2f}')
