from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del request
        data = request.json
        peso = data['peso']
        altura = data['altura']
        edad = data['edad']
        objetivo = data['objetivo']
        sexo = data.get('sexo', '')  # Si se necesita
        
        # Codificar el objetivo
        objetivo_codificado = le_objetivo.transform([objetivo])[0]

        # Crear un array de entrada que incluya todas las características requeridas
        X = np.array([[peso, altura, edad, objetivo_codificado, 0, 0]])
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
        
        # Preparar la respuesta
        response = {
            'desayuno': desayuno_decodificado[0],
            'almuerzo': almuerzo_decodificado[0],
            'merienda': merienda_decodificado[0],
            'cena': cena_decodificado[0],
            'IMC': round(imc, 2),
            'calorias_necesarias': round(calorias_necesarias, 2)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
