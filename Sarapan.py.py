import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import random
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load dataset
dataset = pd.read_csv("Resep_Sarapan.csv")

def hitung_berat_badan_ideal(Tb):
    Bi = (Tb - 100) - (0.1 * (Tb - 100))
    return Bi

def hitung_AKEi_umur(Bi, jenis_kelamin, umur):
    if 20 <= umur <= 29:
        if jenis_kelamin.lower() == "pria":
            AKEi = (15.3 * Bi + 679) * 1.78  
        elif jenis_kelamin.lower() == "wanita":
            AKEi = (14.7 * Bi + 496) * 1.64
        else:
            return None, "Jenis kelamin tidak valid"
    elif 30 <= umur <= 59:
        if jenis_kelamin.lower() == "pria":
            AKEi = (11.6 * Bi + 879) * 1.78  
        elif jenis_kelamin.lower() == "wanita":
            AKEi = (8.7 * Bi + 829) * 1.64
        else:
            return None, "Jenis kelamin tidak valid"
    elif umur >= 60:
        if jenis_kelamin.lower() == "pria":
            AKEi = (13.5 * Bi + 487) * 1.78  
        elif jenis_kelamin.lower() == "wanita": 
            AKEi = (13.5 * Bi + 596) * 1.64
        else:
            return None, "Jenis kelamin tidak valid"
    else:
        return None, "Umur tidak valid"
    return AKEi, None

# Function to calculate nutritional needs based on mealtime and health condition
def hitung_kebutuhan_nutrisi(mealtime, AKEi, penyakit_input_list, jenis_kelamin, alergi):
    faktor = hitung_kebutuhan_faktor(mealtime)
    penyakit_input = set(penyakit_input_list) 

    kebutuhan_kalori = protein = lemak = lemak_jenuh = lemak_tidak_jenuh_ganda = lemak_tidak_jenuh_tunggal = karbohidrat = kolesterol = gula = serat = garam = kalium = 0

    if {'Diabetes', 'Hipertensi', 'Kolesterol'}.issubset(penyakit_input):
        kebutuhan_kalori = faktor * AKEi
        protein = 0.8 * kebutuhan_kalori / 4
        lemak = 0.2 * kebutuhan_kalori / 9
        lemak_jenuh = 0.5 * lemak / 9
        lemak_tidak_jenuh_ganda = 0.1 * lemak
        lemak_tidak_jenuh_tunggal = lemak - lemak_jenuh - lemak_tidak_jenuh_ganda
        karbohidrat = 0.55 * kebutuhan_kalori / 4
        kolesterol = faktor * 200
        gula = 0.025 * kebutuhan_kalori
        serat = faktor * 12.5
        garam = faktor * 1500
        kalium = faktor * 3500
        print("Ini ketiga penyakit disatukan")
        # Kondisi untuk diabetes, hipertensi, dan kolesterol bersamaan
        return np.array([[kebutuhan_kalori, protein, lemak, lemak_jenuh, lemak_tidak_jenuh_ganda, lemak_tidak_jenuh_tunggal, karbohidrat, kolesterol, gula, serat, garam, kalium]])
    
    if {'Diabetes', 'Hipertensi'}.issubset(penyakit_input):
        # Kondisi untuk diabetes dan hipertensi
        kebutuhan_kalori = faktor * AKEi
        protein = 0.8 * kebutuhan_kalori / 4
        lemak = 0.225 * kebutuhan_kalori / 9
        lemak_jenuh = 0.05 * lemak / 9
        lemak_tidak_jenuh_ganda = 0.1 * lemak
        lemak_tidak_jenuh_tunggal = lemak - lemak_jenuh - lemak_tidak_jenuh_ganda
        karbohidrat = 0.55 * kebutuhan_kalori / 4
        kolesterol = faktor * 200
        gula = 0.025 * kebutuhan_kalori
        serat = faktor * 12.5
        garam = faktor * 1500
        kalium = faktor * 3500
        print("Ini diabetes dan hipertensi")
        return np.array([[kebutuhan_kalori, protein, lemak, lemak_jenuh, lemak_tidak_jenuh_ganda, lemak_tidak_jenuh_tunggal, karbohidrat, kolesterol, gula, serat, garam, kalium]])
        
    if {'Diabetes', 'Kolesterol'}.issubset(penyakit_input):
        # Kondisi untuk diabetes dan kolesterol
        kebutuhan_kalori = faktor * AKEi
        protein = 0.8 * kebutuhan_kalori / 4
        lemak = 0.2 * kebutuhan_kalori / 9
        lemak_jenuh = 0.05 * lemak / 9
        lemak_tidak_jenuh_ganda = 0.1 * lemak
        lemak_tidak_jenuh_tunggal = lemak - lemak_jenuh - lemak_tidak_jenuh_ganda
        karbohidrat = 0.55 * kebutuhan_kalori / 4
        kolesterol = faktor * 200
        gula = 0.025 * kebutuhan_kalori
        serat = faktor * 12.5
        garam = faktor * 1500
        kalium = faktor * 3500
        print("Ini diabetes dan kolesterol")
        return np.array([[kebutuhan_kalori, protein, lemak, lemak_jenuh, lemak_tidak_jenuh_ganda, lemak_tidak_jenuh_tunggal, karbohidrat, kolesterol, gula, serat, garam, kalium]])
        
    if {'Hipertensi', 'Kolesterol'}.issubset(penyakit_input):
        # Kondisi untuk hipertensi dan kolesterol
        kebutuhan_kalori = faktor * AKEi
        protein = 0.8 * kebutuhan_kalori / 4
        lemak = 0.2 * kebutuhan_kalori / 9
        lemak_jenuh = 0.05 * lemak / 9
        lemak_tidak_jenuh_ganda = 0.1 * lemak
        lemak_tidak_jenuh_tunggal = lemak - lemak_jenuh - lemak_tidak_jenuh_ganda
        karbohidrat = 0.6 * kebutuhan_kalori / 4
        kolesterol = faktor * 200
        gula = 0.025 * kebutuhan_kalori
        serat = faktor * 12.5
        garam = faktor * 2400
        kalium = faktor * 3500
        print("Ini hipertensi dan kolesterol")
        return np.array([[kebutuhan_kalori, protein, lemak, lemak_jenuh, lemak_tidak_jenuh_ganda, lemak_tidak_jenuh_tunggal, karbohidrat, kolesterol, gula, serat, garam, kalium]])
    
    if 'Diabetes' in penyakit_input:
        kebutuhan_kalori = faktor * AKEi
        protein = 0.125 * kebutuhan_kalori / 4
        lemak = 0.225 * kebutuhan_kalori / 9
        lemak_jenuh = 0.05 * lemak / 9
        lemak_tidak_jenuh_ganda = 0.1 * lemak
        lemak_tidak_jenuh_tunggal = lemak - lemak_jenuh - lemak_tidak_jenuh_ganda
        karbohidrat = 0.65 * kebutuhan_kalori / 4
        kolesterol = faktor * 150
        gula = 0.025 * kebutuhan_kalori
        serat = faktor * 12.5
        garam = faktor * 1500
        kalium = faktor * 3500
        print("Ini diabetes")
    elif 'Hipertensi' in penyakit_input:
        kebutuhan_kalori = faktor * AKEi
        protein = 0.8 * kebutuhan_kalori / 4
        lemak = 0.25 * kebutuhan_kalori / 9
        lemak_jenuh = 0.07 * kebutuhan_kalori / 9
        lemak_tidak_jenuh_ganda = 0.1 * lemak
        lemak_tidak_jenuh_tunggal = lemak - lemak_jenuh - lemak_tidak_jenuh_ganda
        karbohidrat = 0.625 * kebutuhan_kalori / 4
        kolesterol = 300
        gula = 0.025 * kebutuhan_kalori
        serat = 12.5
        garam = 2400
        kalium = 3500
        print("Ini hipertensi")
    elif 'Kolesterol' in penyakit_input:
        kebutuhan_kalori = faktor * AKEi
        protein = 0.8 * AKEi / 4
        karbohidrat = 0.65 * AKEi / 4
        lemak = 0.225 * AKEi / 9
        lemak_jenuh = 0.07 * AKEi / 9
        lemak_tidak_jenuh_ganda = 0.1 * lemak
        lemak_tidak_jenuh_tunggal = lemak - lemak_jenuh - lemak_tidak_jenuh_ganda
        kolesterol = 200
        gula = 0.025 * AKEi
        if jenis_kelamin == 'pria':
            serat = 38
        elif jenis_kelamin.lower() == 'wanita':
            serat = 25
        else:
           raise ValueError("Jenis kelamin tidak valid") 
        garam = 2400
        kalium = 3500
        print("Ini kolesterol")
    else:
        return ValueError("Penyakit tidak valid")
    
    return np.array([[kebutuhan_kalori, protein, lemak, lemak_jenuh, lemak_tidak_jenuh_ganda, lemak_tidak_jenuh_tunggal, karbohidrat, kolesterol, gula, serat, garam, kalium]])

# Function to calculate the scaling factor for nutritional needs based on mealtime
def hitung_kebutuhan_faktor(mealtime):
    if mealtime == "makan pagi":
        faktor = 0.20
    elif mealtime == "selingan makan pagi":
        faktor = 0.10
    else:
        raise ValueError("Jam tidak valid.")
    return faktor

@app.route('/recommend', methods=['POST'])
def recommend():
    global dataset  # Add this line to declare 'dataset' as a global variable within this function
    data = request.get_json()
    if not data or 'Tinggi Badan' not in data or 'Jenis Kelamin' not in data or 'Umur' not in data:
        return jsonify({"error": "Data input tidak lengkap"}), 400

    Tb = data['Tinggi Badan']
    jenis_kelamin = data['Jenis Kelamin']
    umur = data['Umur']
    penyakit_input = data.get('Penyakit', '').split(",")
    alergi = data.get('Alergi', '').split(",")
    mode = data.get('Mode')  # Corrected from data.get['Mode'] to data.get('Mode')

    # Handle the possibility of 'mode' being None
    if mode is not None:
        mode = mode.lower()
        if mode == "daily":
            num_recommendations = 1
        elif mode == "weekly":
            num_recommendations = 7
        else:
            return jsonify({"error": "Invalid mode specified"}), 400
    else:
        return jsonify({"error": "Mode not specified"}), 400
    
    k = 7  # Number of nearest neighbors to consider

    # Shuffle the dataset to ensure different foods are recommended for each mealtime
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    berat_badan_ideal = hitung_berat_badan_ideal(Tb)
    AKEi, error = hitung_AKEi_umur(berat_badan_ideal, jenis_kelamin, umur)
    if error:
        return jsonify({"error": error}), 400

    target = hitung_kebutuhan_nutrisi("makan pagi", AKEi, penyakit_input, jenis_kelamin, alergi)
    if target is None:
        return jsonify({"error": "Error calculating nutritional needs"}), 500

    features = dataset[['Energi (kkal)', 'Protein (g)', 'Lemak (g)', 'Lemak Jenuh (g)', 'Lemak tak Jenuh Ganda (g)', 'Lemak tak Jenuh Tunggal (g)', 'Karbohidrat (g)', 'Kolesterol (mg)', 'Gula (g)', 'Serat (g)', 'Sodium (mg)', 'Kalium (mg)']]
    knn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    knn.fit(features)
    
    for j in range(num_recommendations):
        distances, indices = knn.kneighbors(target, return_distance=True)
        idx = indices[0][j]

        # Retrieve the food item from the dataset
        recommended_food = dataset.loc[idx]

        # If the user has allergies, check if the recommended food contains any allergen
        if alergi:
            contains_allergen = False
            for allergen in alergi:
                if allergen in recommended_food['Ingredients']:
                    contains_allergen = True
                    break

            # If the recommended food contains an allergen, select a different one
            if contains_allergen:
                # Find a food item that does not contain allergens
                non_allergen_indices = dataset[~dataset['Ingredients'].str.contains('|'.join(alergi), na=False)].index
                non_allergen_idx = random.choice(non_allergen_indices)
                recommended_food = dataset.loc[non_allergen_idx]
                 
    results = []
    num_available_recommendations = len(indices[0])  # Get the number of available recommendations

    # Ensure the loop only iterates up to the minimum of num_recommendations or num_available_recommendations
    for j in range(min(num_recommendations, num_available_recommendations)):
        idx = indices[0][j]
        recommended_food = dataset.iloc[idx]
        if alergi:  # Check for allergies if applicable
            contains_allergen = any(allergen in recommended_food['Ingredients'] for allergen in alergi)
            if contains_allergen:
                continue  # Skip this recommendation if it contains allergens
        results.append({
            "Nama Resep": str(recommended_food['Nama Resep']),
            "Energi": int(recommended_food['Energi (kkal)']),
            "Karbohidrat": float(recommended_food['Karbohidrat (g)']),
            "Lemak": float(recommended_food['Lemak (g)']),
            "Protein": float(recommended_food['Protein (g)'])
        })

    # Return results, potentially fewer than requested if limited by dataset size or allergen constraints
    return jsonify(results)

if __name__ == "__main__":
    app.run(port=8080, debug=True)

