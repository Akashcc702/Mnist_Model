// ===== GLOBALS =====
let model;
let canvas, ctx;
let drawing = false;

// ===== INIT =====
async function init() {
    console.log("Loading model...");

    model = await tf.loadLayersModel("model/mnist_model.json");

    console.log("Model Loaded ✅");

    canvas = document.getElementById("sketchpad");
    ctx = canvas.getContext("2d");

    // white background
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // mouse events
    canvas.addEventListener("mousedown", startDraw);
    canvas.addEventListener("mouseup", stopDraw);
    canvas.addEventListener("mousemove", draw);

    // touch support
    canvas.addEventListener("touchstart", startDraw);
    canvas.addEventListener("touchend", stopDraw);
    canvas.addEventListener("touchmove", draw);

    // buttons
    document.getElementById("predict_button").addEventListener("click", predict);
    document.getElementById("clear_button").addEventListener("click", clearCanvas);
}

// ===== DRAW =====
function startDraw(e) {
    drawing = true;
}

function stopDraw() {
    drawing = false;
    ctx.beginPath();
}

function draw(e) {
    if (!drawing) return;

    let rect = canvas.getBoundingClientRect();
    let x = (e.clientX || e.touches[0].clientX) - rect.left;
    let y = (e.clientY || e.touches[0].clientY) - rect.top;

    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// ===== CLEAR =====
function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    document.getElementById("result").innerText = "-";
    document.getElementById("confidence").innerText = "Confidence: -";
}

// ===== PREPROCESS =====
function preprocessCanvas() {
    let tempCanvas = document.createElement("canvas");
    let tempCtx = tempCanvas.getContext("2d");

    tempCanvas.width = 28;
    tempCanvas.height = 28;

    // resize
    tempCtx.drawImage(canvas, 0, 0, 28, 28);

    let imgData = tempCtx.getImageData(0, 0, 28, 28);
    let data = imgData.data;

    let input = [];

    for (let i = 0; i < data.length; i += 4) {
        // take only one channel (grayscale)
        let avg = data[i]; 
        input.push(avg / 255);
    }

    return tf.tensor(input).reshape([1, 28, 28, 1]);
}

// ===== PREDICT =====
async function predict() {
    let input = preprocessCanvas();

    let prediction = model.predict(input);
    let probs = prediction.dataSync();

    let max = Math.max(...probs);
    let result = probs.indexOf(max);

    document.getElementById("result").innerText = result;
    document.getElementById("confidence").innerText =
        "Confidence: " + (max * 100).toFixed(2) + "%";
}
