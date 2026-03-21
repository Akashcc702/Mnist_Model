// ===== GLOBALS =====
let model;
let canvas, ctx;
let drawing = false;

// ===== INIT =====
async function init() {
    console.log("INIT CALLED");

    try {
        // 🔥 Loading indicator
        document.getElementById("result").innerText = "Loading...";

        // ✅ Load model (CDN)
        model = await tf.loadLayersModel(
            "https://cdn.jsdelivr.net/gh/Akashcc702/Mnist_Model/model/mnist_model.json"
        );

        console.log("Model Loaded ✅");

        document.getElementById("result").innerText = "Ready";

    } catch (error) {
        console.error("Model Load Error:", error);
        alert("Model failed to load ❌");
        return;
    }

    canvas = document.getElementById("sketchpad");
    ctx = canvas.getContext("2d");

    // black background
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // ===== EVENTS =====

    // mouse
    canvas.addEventListener("mousedown", () => drawing = true);
    canvas.addEventListener("mouseup", stopDraw);
    canvas.addEventListener("mousemove", draw);

    // touch (mobile fix)
    canvas.addEventListener("touchstart", (e) => {
        e.preventDefault();
        drawing = true;
    });

    canvas.addEventListener("touchend", stopDraw);

    canvas.addEventListener("touchmove", (e) => {
        e.preventDefault();
        draw(e);
    });

    // buttons
    document.getElementById("predict_button").addEventListener("click", predict);
    document.getElementById("clear_button").addEventListener("click", clearCanvas);
}

// ===== DRAW =====
function stopDraw() {
    drawing = false;
    ctx.beginPath();
}

function draw(e) {
    if (!drawing) return;

    let rect = canvas.getBoundingClientRect();
    let x, y;

    if (e.touches && e.touches.length > 0) {
        x = e.touches[0].clientX - rect.left;
        y = e.touches[0].clientY - rect.top;
    } else {
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
    }

    ctx.lineWidth = 18;
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
        let pixel = data[i]; // grayscale
        input.push(pixel / 255);
    }

    return tf.tensor(input).reshape([1, 28, 28, 1]);
}

// ===== PREDICT =====
async function predict() {
    if (!model) {
        alert("Model not loaded yet ❌");
        return;
    }

    document.getElementById("result").innerText = "...";

    const input = preprocessCanvas();

    const prediction = model.predict(input);
    const probs = prediction.dataSync();

    const max = Math.max(...probs);
    const result = probs.indexOf(max);

    document.getElementById("result").innerText = result;
    document.getElementById("confidence").innerText =
        "Confidence: " + (max * 100).toFixed(2) + "%";

    // cleanup
    tf.dispose([input, prediction]);
}
