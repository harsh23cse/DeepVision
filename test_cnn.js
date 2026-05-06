const tf = require('@tensorflow/tfjs-node');

async function run() {
    const model = await tf.loadLayersModel('https://raw.githubusercontent.com/anshsaini/Digit-Recognizer/master/model.json');
    console.log("Model loaded!");
    model.summary();
}
run();
