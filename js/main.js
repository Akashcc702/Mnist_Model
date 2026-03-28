// ===== GLOBALS =====
let model;
let canvas, ctx;
let drawing = false;
let chart = null;

// ===== INIT =====
async function init() {
    console.log("INIT CALLED");

    try {
        document.getElementById("result").innerText = "Loading...";

        model = await tf.loadLayersModel("./model/mnist_model.json");

        console.log("Model Loaded ✅");
        document.getElementById("result").innerText = "Ready";

    } catch (error) {
        console.error("Model Load Error:", error);
        alert("Model failed to load ❌");
        return;
    }

    canvas = document.getElementById("sketchpad");
    ctx = canvas.getContext("2d");

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // EVENTS
    canvas.addEventListener("mousedown", () => drawing = true);
    canvas.addEventListener("mouseup", stopDraw);
    canvas.addEventListener("mousemove", draw);

    canvas.addEventListener("touchstart", (e) => {
        e.preventDefault();
        drawing = true;
    });

    canvas.addEventListener("touchend", stopDraw);

    canvas.addEventListener("touchmove", (e) => {
        e.preventDefault();
        draw(e);
    });

    document.getElementById("predict_button").addEventListener("click", predict);
    document.getElementById("clear_button").addEventListener("click", clearCanvas);

    speechSynthesis.getVoices();
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

    if (chart) {
        chart.data.datasets[0].data = [0,0,0,0,0,0,0,0,0,0];
        chart.update('none'); // ❌ no animation
    }
}

// ===== PREPROCESS =====
function preprocessCanvas() {
    let tempCanvas = document.createElement("canvas");
    let tempCtx = tempCanvas.getContext("2d");

    tempCanvas.width = 28;
    tempCanvas.height = 28;

    tempCtx.drawImage(canvas, 0, 0, 28, 28);

    let imgData = tempCtx.getImageData(0, 0, 28, 28);
    let data = imgData.data;

    let input = [];

    for (let i = 0; i < data.length; i += 4) {
        let pixel = data[i];
        input.push(pixel / 255);
    }

    return tf.tensor(input).reshape([1, 28, 28, 1]);
}

// ===== GRAPH CREATE =====
function createGraph(data) {
    const canvasEl = document.getElementById("myChart");
    if (!canvasEl) return;

    const ctxChart = canvasEl.getContext("2d");

    if (chart) {
        chart.destroy();
        chart = null;
    }

    chart = new Chart(ctxChart, {
        type: 'line', // ✅ LINE CHART
        data: {
            labels: ['0','1','2','3','4','5','6','7','8','9'],
            datasets: [{
                label: 'Confidence',
                data: data,
                fill: false,
                tension: 0.3 // smooth curve
            }]
        },
        options: {
            responsive: true,
            animation: false, // ❌ STOP RUNNING
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

// ===== GRAPH UPDATE =====
function updateGraph(data) {
    if (!chart) {
        createGraph(data);
        return;
    }

    chart.data.datasets[0].data = data;
    chart.update('none'); // ❌ NO ANIMATION
}

// ===== SPEECH =====
function speakNumber(number) {
    const toggle = document.getElementById("voice_toggle");
    if (toggle && !toggle.checked) return;

    const langEl = document.getElementById("language_select");
    const lang = langEl ? langEl.value : "en";

    let text = "";
    let languageCode = "en-US";

    if (lang === "kn") {
        text = "ನೀವು ಬರೆದ ಸಂಕೆ " + number;
        languageCode = "kn-IN";
    } else {
        text = "The predicted number is " + number;
    }

    const msg = new SpeechSynthesisUtterance(text);
    msg.lang = languageCode;
    msg.rate = 0.9;

    const voices = speechSynthesis.getVoices();
    const voice =
        voices.find(v => v.lang === languageCode) ||
        voices.find(v => v.lang.includes("en")) ||
        voices[0];

    if (voice) msg.voice = voice;

    speechSynthesis.cancel();
    speechSynthesis.speak(msg);
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

    updateGraph(probs);

    speakNumber(result);

    tf.dispose([input, prediction]);
}
