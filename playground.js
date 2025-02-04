// Import TensorFlow.js
// import * as tf from '@tensorflow/tfjs';

// HTML Elements setup (assume this is part of your HTML):
// <input type="file" id="fileInput" accept=".csv" />
// <div id="dataPreview"></div>
// <button id="trainModel">Train Model</button>
// <div id="metricsOutput"></div>

// Function to read CSV and parse data
async function loadCSVData(file) {
    const reader = new FileReader();
    return new Promise((resolve, reject) => {
        reader.onload = () => {
            const text = reader.result;
            const data = tf.data.csv(text, { hasHeader: true });
            resolve(data);
        };
        reader.onerror = reject;
        reader.readAsText(file);
    });
}

// Function to build a simple ANN model
function buildModel(inputShape) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [inputShape] }));
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' })); // For binary classification
    model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
    return model;
}

// Function to train the model
async function trainModel(model, data) {
    const processedData = data.map(row => ({
        xs: Object.values(row).slice(0, -1),
        ys: [Object.values(row).slice(-1)[0]]
    }));
    const dataset = tf.data.array(processedData).batch(32);

    const history = await model.fitDataset(dataset, {
        epochs: 10,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                document.getElementById('metricsOutput').innerText = 
                    `Epoch: ${epoch + 1}, Loss: ${logs.loss.toFixed(4)}, Accuracy: ${logs.acc.toFixed(4)}`;
            }
        }
    });
}

// Event listeners
const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        const data = await loadCSVData(file);
        document.getElementById('dataPreview').innerText = 'Data loaded successfully';

        // Build and train the model
        const model = buildModel(data.elementSpec.shape[0]);
        document.getElementById('trainModel').addEventListener('click', () => trainModel(model, data));
    }
});
