from flask import Flask, Response, request, jsonify
import pandas as pd
import json
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

count = 0
numberOfUsers = 10
version = 0

def federative_averages(gradients):
    gradients_array = np.array(gradients)
    
    averages = np.mean(gradients_array, axis=0)

    return averages

# Function to update item vector
def update_item_vector(item_vector, gradient):
    item_vectors = np.copy(item_vector)
    for i in range(len(item_vectors)):
        for j in range(len(item_vectors[i])):
            item_vectors[i][j] = item_vectors[i][j] - gradient[i][j]
    return item_vectors

@app.route('/version', methods=['GET'])
def get_version():
    global version
    # Return JSON response
    return Response(str(version))

@app.route('/items', methods=['GET'])
def get_items():
    df = pd.read_csv("hotelmatrix.csv")
    csv_string = df.to_csv(index=False, header=False)

    # Remove trailing line break
    csv_string = csv_string.strip()

    # Return JSON response
    return Response(csv_string)

@app.route('/storematrix', methods=['POST'])
def store_matrix():
    global count
    global version
    # Get JSON data from request
    matrix_data = request.get_json()

    # Convert list of lists to DataFrame
    df = pd.DataFrame(matrix_data)
    # Save DataFrame to a CSV file
    df.to_csv(f"Gradients/gradient{count}.csv", index=False, header=None)
    count = count + 1
    if count == numberOfUsers:
        print("recomputing model")
        item_vector = pd.read_csv("hotelmatrix.csv")
        item_id_list = item_vector["hotel_code"]
        item_vector = item_vector.apply(pd.to_numeric, errors='coerce').iloc[:, 1:].to_numpy()
        gradients = []
        for i in range(numberOfUsers):
           gradient = pd.read_csv(f"Gradients/gradient{i}.csv",header= None).apply(pd.to_numeric, errors='coerce').to_numpy()
           gradients.append(gradient)
        global_gradient = federative_averages(gradients)
        new_item_vector = update_item_vector(item_vector, global_gradient)
        new_item_vector_df = pd.DataFrame(new_item_vector)
        new_item_vector_df.insert(0,"hotel_code",item_id_list)
        new_item_vector_df.to_csv("hotelmatrix.csv",index=False)
        version = version + 1
        count = 0
    return jsonify({'message': 'Matrix successfully stored'}), 200

if __name__ == "__main__":
    app.run(port=8000)

