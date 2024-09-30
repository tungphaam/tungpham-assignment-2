from flask import Flask, render_template, request, jsonify
import numpy as np
from kmeans import KMeans  # Import your KMeans class

app = Flask(__name__)

@app.route('/')
def index():
    # Renders the main KMeans Clustering page
    return render_template('index.html')

@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    global current_dataset, current_centroids
    try:
        data = request.json
        k = int(data['k'])
        init_method = data['init_method']

        if current_dataset is None:
            current_dataset = np.array(data['data'])
        
        if 'centroids' in data and data['centroids']:
            current_centroids = np.array(data['centroids'])
        else:
            current_centroids = None  
        
        # Create KMeans object with the selected initialization method and existing centroids
        kmeans = KMeans(k=k, init_method=init_method, initial_centroids=current_centroids)
        kmeans.fit(current_dataset) 
        current_centroids = kmeans.centroids
        
        # Return centroids and clusters
        centroids = kmeans.centroids.tolist()
        clusters = kmeans.predict(current_dataset).tolist()
        
        return jsonify({'centroids': centroids, 'clusters': clusters, 'data': current_dataset.tolist()})
    
    except Exception as e:
        print(f"Error during KMeans execution: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_new_dataset', methods=['POST'])
def generate_new_dataset():
    global current_dataset, current_centroids
    try:
        # Generates a new random dataset
        current_dataset = np.random.uniform(-10, 10, (300, 2))
        current_centroids = None  
        return jsonify({'data': current_dataset.tolist()})
    
    except Exception as e:
        print(f"Error generating new dataset: {e}")
        return jsonify({'error': f"Error: {str(e)}"}), 500


@app.route('/step_through_kmeans', methods=['POST'])
def step_through_kmeans():
    try:
        data = request.json
        k = int(data['k'])
        init_method = data['init_method']

        # If the dataset is passed in the request, use it; otherwise, generate one
        if 'data' in data and data['data']:
            current_dataset = np.array(data['data'])
        else:
            # Generate random dataset if no dataset is provided
            current_dataset = np.random.uniform(-10, 10, (300, 2))

        # Run KMeans and store steps
        kmeans = KMeans(k=k, init_method=init_method)
        kmeans.fit(current_dataset)

        # Convert Numpy arrays to lists before returning them as JSON
        steps = kmeans.get_steps()
        steps_serializable = []
        for step in steps:
            steps_serializable.append({
                'centroids': step['centroids'].tolist(),
                'clusters': step['clusters'].tolist()
            })

        return jsonify({'steps': steps_serializable, 'data': current_dataset.tolist()})
    
    except Exception as e:
        print(f"Error during KMeans step-through: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/run_manual_kmeans', methods=['POST'])
def run_manual_kmeans():
    global current_dataset, current_centroids
    try:
        data = request.json
        manual_centroids = np.array(data['centroids'])  
        dataset = np.array(data['data']) 
        current_dataset = dataset  
        
        # Create a KMeans object with manually selected centroids
        kmeans = KMeans(k=len(manual_centroids), init_method="manual")
        kmeans.centroids = manual_centroids  
        
        # Run KMeans with the dataset
        kmeans.fit(dataset)

        # Collect step-by-step information
        steps = kmeans.get_steps()  
        
        steps_serializable = []
        for step in steps:
            steps_serializable.append({
                'centroids': step['centroids'].tolist(),
                'clusters': step['clusters'].tolist()
            })

        return jsonify({'steps': steps_serializable, 'data': dataset.tolist()})

    except Exception as e:
        print(f"Error during manual KMeans execution: {e}")
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, port=3000)
