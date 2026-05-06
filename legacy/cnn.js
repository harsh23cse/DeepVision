/* ============================================================
   CNN Learning Explainer — Interactions & Visualizations
   ============================================================ */
document.addEventListener('DOMContentLoaded', () => {

    // ===== SCROLL REVEAL =====
    const reveals = document.querySelectorAll('.section, .feature-card, .arch-block, .weight-panel, .rw-step');
    reveals.forEach(el => el.classList.add('reveal'));
    const obs = new IntersectionObserver(entries => {
        entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); obs.unobserve(e.target); } });
    }, { threshold: 0.1 });
    reveals.forEach(el => obs.observe(el));


    // ===== HERO ANIMATION =====
    function initHeroCNN() {
        const canvas = document.getElementById('hero-cnn-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        
        function resize() {
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight || 300;
        }
        window.addEventListener('resize', resize);
        resize();
        
        let t = 0;
        function draw() {
            t += 0.03;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const w = canvas.width;
            const h = canvas.height;
            const layers = [10, 6, 4, 2];
            
            ctx.lineWidth = 1;
            for (let i = 0; i < layers.length - 1; i++) {
                const count1 = layers[i];
                const count2 = layers[i+1];
                const x1 = w * (0.2 + i * 0.2);
                const x2 = w * (0.2 + (i + 1) * 0.2);
                
                for (let j = 0; j < count1; j++) {
                    const y1 = h * ((j + 1) / (count1 + 1));
                    for (let k = 0; k < count2; k++) {
                        const y2 = h * ((k + 1) / (count2 + 1));
                        
                        ctx.beginPath();
                        ctx.moveTo(x1, y1);
                        ctx.bezierCurveTo(x1 + 50, y1, x2 - 50, y2, x2, y2);
                        
                        const alpha = Math.max(0.05, 0.2 * Math.sin(t + j + k));
                        ctx.strokeStyle = `rgba(99, 102, 241, ${alpha})`;
                        ctx.stroke();
                        
                        // Draw animated particles moving along connection
                        const progress = (t * 0.5 + j * 0.1 + k * 0.2) % 1;
                        const px = x1 + (x2 - x1) * progress;
                        // Linear interpolation for Y since bezier is hard to map without getPointAtLength
                        const py = y1 + (y2 - y1) * progress; 
                        
                        if (alpha > 0.1) {
                            ctx.fillStyle = `rgba(16, 185, 129, ${alpha * 2})`;
                            ctx.beginPath();
                            ctx.arc(px, py, 1.5, 0, Math.PI * 2);
                            ctx.fill();
                        }
                    }
                    
                    ctx.beginPath();
                    ctx.arc(x1, y1, 4 + Math.sin(t * 2 + j), 0, Math.PI * 2);
                    ctx.fillStyle = '#6366f1';
                    ctx.fill();
                }
            }
            
            requestAnimationFrame(draw);
        }
        draw();
    }
    initHeroCNN();

    // ===== TF.JS IMPLEMENTATION FOR VISUAL LEARNING =====
    
    // UI Elements
    const btnNextStep = document.getElementById('btn-next-step');
    
    // State
    let model = null;
    let isModelLoaded = false;
    let stepState = 0; 
    let selectedInputIdx = 0; 
    
    // Load MobileNet
    async function loadRealTimeModel() {
        try {
            console.log('Loading MobileNet...');
            model = await mobilenet.load({ version: 2, alpha: 1.0 });
            isModelLoaded = true;
            console.log('MobileNet loaded.');
        } catch (e) {
            console.error('Failed to load MobileNet', e);
        }
    }
    loadRealTimeModel();

    // Layer Inspector Logic
    let activeLayer = null;
    const inspectorContent = document.getElementById('inspector-content');
    const inspectorTitle = document.getElementById('inspector-title');
    const fmapGrid = document.getElementById('inspector-fmaps');
    const weightGrid = document.getElementById('inspector-weights');
    
    document.querySelectorAll('.net-node').forEach(node => {
        node.addEventListener('click', () => {
            // Remove active from all
            document.querySelectorAll('.net-node').forEach(n => n.classList.remove('active'));
            node.classList.add('active');
            
            const layerName = node.getAttribute('data-layer');
            activeLayer = layerName;
            inspectorTitle.textContent = `Inspecting: ${node.querySelector('.node-label').textContent}`;
            inspectorContent.style.display = 'flex';
            document.querySelector('.inspector-hint').style.display = 'none';
            
            inspectLayer(layerName);
        });
    });

    function inspectLayer(layerName) {
        if (!model) return;
        
        // Try to get layer
        let layer;
        try {
            layer = model.getLayer(layerName);
        } catch (e) {
            // Might be input layer which is implicit
            layer = null;
        }

        fmapGrid.innerHTML = '';
        weightGrid.innerHTML = '';

        if (!layer) return; // e.g. input layer

        // Visualize Weights if it's a Convolutional Layer
        if (layer.getClassName() === 'Conv2D') {
            const weights = layer.getWeights()[0]; // Shape: [kernelSize, kernelSize, inChannels, outChannels]
            if (weights) {
                const wData = weights.dataSync();
                const outChannels = weights.shape[3];
                const kernelSize = weights.shape[0]; // assuming square
                
                // Show up to 4 filters
                const numToShow = Math.min(outChannels, 4);
                
                for (let i = 0; i < numToShow; i++) {
                    const canvas = document.createElement('canvas');
                    canvas.width = 40; canvas.height = 40;
                    const ctx = canvas.getContext('2d');
                    const imgData = ctx.createImageData(40, 40);
                    
                    // Determine min max for scaling
                    let minW = Infinity, maxW = -Infinity;
                    for(let k=0; k<wData.length; k++) {
                        if(wData[k] < minW) minW = wData[k];
                        if(wData[k] > maxW) maxW = wData[k];
                    }
                    const rangeW = Math.max(Math.abs(maxW), Math.abs(minW)) || 1;

                    for (let y = 0; y < 40; y++) {
                        for (let x = 0; x < 40; x++) {
                            const ky = Math.floor((y / 40) * kernelSize);
                            const kx = Math.floor((x / 40) * kernelSize);
                            
                            // Get weight for the first input channel for simplicity
                            const val = weights.arraySync()[ky][kx][0][i];
                            const normalized = val / rangeW; // -1 to 1
                            
                            let r, g, b;
                            if (normalized < 0) {
                                // Blue for negative
                                r = g = Math.floor(255 * (1 + normalized));
                                b = 255;
                            } else {
                                // Red for positive
                                r = 255;
                                g = b = Math.floor(255 * (1 - normalized));
                            }
                            
                            const idx = (y * 40 + x) * 4;
                            imgData.data[idx] = r;
                            imgData.data[idx+1] = g;
                            imgData.data[idx+2] = b;
                            imgData.data[idx+3] = 255;
                        }
                    }
                    ctx.putImageData(imgData, 0, 0);
                    weightGrid.appendChild(canvas);
                }
            }
        }
    }

    // Step-by-Step Animation Engine
    const stepDesc = document.getElementById('step-desc');
    const skipPath = document.getElementById('res-skip-path');
    
    if (btnNextStep) {
        btnNextStep.addEventListener('click', () => {
            stepState = (stepState + 1) % 5;
            
            // Reset styling
            document.querySelectorAll('.net-node').forEach(n => n.style.opacity = '0.5');
            skipPath.style.borderColor = 'rgba(99,102,241,0.2)';
            skipPath.style.boxShadow = 'none';
            
            switch (stepState) {
                case 1:
                    btnNextStep.textContent = "Next Step: Feature Extraction";
                    stepDesc.innerHTML = "<strong>Forward Pass:</strong> Data enters the network and hits the first Convolutional Layer, where basic edges are detected.";
                    document.querySelector('[data-layer="input"]').style.opacity = '1';
                    document.querySelector('[data-layer="conv1"]').style.opacity = '1';
                    break;
                case 2:
                    btnNextStep.textContent = "Next Step: Skip Connection";
                    stepDesc.innerHTML = "<strong>Feature Extraction:</strong> Data is pooled to reduce dimensions and sent to the Residual Block.";
                    document.querySelector('[data-layer="pool1"]').style.opacity = '1';
                    document.querySelector('[data-layer="res_conv"]').style.opacity = '1';
                    break;
                case 3:
                    btnNextStep.textContent = "Next Step: Backpropagation";
                    stepDesc.innerHTML = "<strong>Skip Connection:</strong> Crucial step! The original pooled features bypass the block and are added directly to the output. This prevents vanishing gradients.";
                    document.querySelector('[data-layer="res_add"]').style.opacity = '1';
                    skipPath.style.borderColor = 'var(--green)';
                    skipPath.style.boxShadow = '0 0 15px var(--green)';
                    break;
                case 4:
                    btnNextStep.textContent = "Reset Explorer";
                    stepDesc.innerHTML = "<strong>Backpropagation:</strong> The model calculates the loss at the Output layer and sends errors backwards (red animation) to update the weights.";
                    document.querySelector('[data-layer="output"]').style.opacity = '1';
                    // Trigger a CSS animation class
                    document.querySelector('.network-container').classList.add('backprop-anim');
                    setTimeout(() => {
                        document.querySelector('.network-container').classList.remove('backprop-anim');
                    }, 1000);
                    break;
                case 0:
                    btnNextStep.textContent = "Next Step: Forward Pass";
                    stepDesc.innerHTML = "Click 'Next Step' to see how data flows through the network.";
                    document.querySelectorAll('.net-node').forEach(n => n.style.opacity = '1');
                    break;
            }
        });
    }

    // ===== STEP 1 & 2: INPUT IMAGE & CONVOLUTIONS =====
    const uploadZone = document.getElementById('upload-zone');
    const imageUploadInput = document.getElementById('image-upload-input');
    const previewZone = document.getElementById('preview-zone');
    const inputPreviewCanvas = document.getElementById('input-preview-canvas');
    const previewMetadata = document.getElementById('preview-metadata');
    const btnSendModel = document.getElementById('btn-send-model');
    
    let currentUploadedImage = null;

    if (uploadZone && imageUploadInput) {
        uploadZone.addEventListener('click', () => {
            imageUploadInput.click();
        });

        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleImageUpload(e.dataTransfer.files[0]);
            }
        });

        imageUploadInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleImageUpload(e.target.files[0]);
            }
        });
        
        btnSendModel.addEventListener('click', async () => {
            if (!currentUploadedImage) return;

            if (!isModelLoaded) {
                btnSendModel.textContent = '⏳ Waiting for model to load...';
                return;
            }
            
            // Run real inference
            btnSendModel.disabled = true;
            btnSendModel.textContent = '⏳ Processing...';
            
            let predictions = [];
            try {
                predictions = await model.classify(currentUploadedImage);
            } catch (e) {
                console.error("Classification error:", e);
            }
            
            window.latestPredictions = predictions;

            const connector = document.getElementById('step-connector-1-2');
            const step2 = document.getElementById('step-2');
            const summary = document.getElementById('conv-summary');

            // Reset everything for fresh animation
            connector.style.display = 'flex';
            step2.style.display = 'none';
            step2.classList.remove('active-step');
            if (summary) summary.style.display = 'none';

            // Reset conv items
            document.querySelectorAll('.conv-item').forEach(el => {
                el.classList.remove('conv-revealed');
            });
            document.querySelectorAll('.scan-overlay').forEach(el => {
                el.classList.remove('scanning', 'done');
                el.style.opacity = '1';
            });
            document.querySelectorAll('.conv-canvas-wrapper').forEach(el => {
                el.classList.remove('glow-active');
            });
            // Reset intensity bars
            ['vertical', 'horizontal', 'combined'].forEach(key => {
                const fill = document.getElementById('intensity-' + key);
                const val = document.getElementById('intensity-val-' + key);
                if (fill) fill.style.width = '0';
                if (val) val.textContent = '0%';
            });

            // Scroll to connector
            setTimeout(() => {
                connector.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 50);

            // After a brief delay, show step 2 with animation
            setTimeout(() => {
                btnSendModel.disabled = false;
                btnSendModel.textContent = 'Send to Model 👉';
                step2.style.display = 'block';
                requestAnimationFrame(() => {
                    step2.classList.add('active-step');
                });
                // Run the convolution and trigger staggered reveal
                runInitialConvolution(currentUploadedImage);
                initScanner(currentUploadedImage);

                setTimeout(() => {
                    step2.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }, 100);
            }, 600);
        });

    }



    function handleImageUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                currentUploadedImage = img;
                uploadZone.style.display = 'none';
                previewZone.style.display = 'flex';

                const ctx = inputPreviewCanvas.getContext('2d');
                inputPreviewCanvas.width = img.width;
                inputPreviewCanvas.height = img.height;
                ctx.drawImage(img, 0, 0);

                const channels = 3; 

                previewMetadata.innerHTML = `
                    <div class="meta-row">
                        <span class="meta-label">Size:</span>
                        <span class="meta-value">${img.width} × ${img.height} px</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label">Channels:</span>
                        <span class="meta-value">RGB (${channels})</span>
                    </div>
                `;

                btnSendModel.disabled = false;
                
                // If step 2 is already open, instantly update it
                if (document.getElementById('step-2').style.display === 'block') {
                    runInitialConvolution(img);
                    initScanner(img);
                }
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    function runInitialConvolution(img) {
        const origCanvas = document.getElementById('conv-canvas-original');
        const vCanvas = document.getElementById('conv-canvas-vertical');
        const hCanvas = document.getElementById('conv-canvas-horizontal');
        const cCanvas = document.getElementById('conv-canvas-combined');
        
        // Use higher resolution for better visual quality
        const W = 128; const H = 128;
        [origCanvas, vCanvas, hCanvas, cCanvas].forEach(c => { c.width = W; c.height = H; });
        
        const ctxOrig = origCanvas.getContext('2d');
        ctxOrig.drawImage(img, 0, 0, W, H);
        
        const imgData = ctxOrig.getImageData(0, 0, W, H);
        const data = imgData.data;
        
        // Convert to grayscale float array
        const gray = new Float32Array(W * H);
        for (let i = 0; i < W * H; i++) {
            gray[i] = (data[i*4] * 0.299 + data[i*4+1] * 0.587 + data[i*4+2] * 0.114);
        }
        
        // Sobel Kernels
        const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1]; // Horizontal edges
        const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1]; // Vertical edges
        
        const vEdges = new Float32Array(W * H);
        const hEdges = new Float32Array(W * H);
        const cEdges = new Float32Array(W * H);
        
        let maxV = 0, maxH = 0, maxC = 0;
        let sumV = 0, sumH = 0, sumC = 0;
        let pixelCount = 0;
        
        for (let y = 1; y < H - 1; y++) {
            for (let x = 1; x < W - 1; x++) {
                let sumX = 0, sumY = 0;
                let k = 0;
                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                        const val = gray[(y + ky) * W + (x + kx)];
                        sumX += val * sobelX[k];
                        sumY += val * sobelY[k];
                        k++;
                    }
                }
                const idx = y * W + x;
                vEdges[idx] = Math.abs(sumX);
                hEdges[idx] = Math.abs(sumY);
                cEdges[idx] = Math.sqrt(sumX*sumX + sumY*sumY);
                
                if (vEdges[idx] > maxV) maxV = vEdges[idx];
                if (hEdges[idx] > maxH) maxH = hEdges[idx];
                if (cEdges[idx] > maxC) maxC = cEdges[idx];
                
                sumV += vEdges[idx];
                sumH += hEdges[idx];
                sumC += cEdges[idx];
                pixelCount++;
            }
        }
        
        // Calculate activation intensity as percentage of max theoretical response
        const maxTheoretical = 255 * 4; // Sobel max response
        const intensityV = Math.min(100, Math.round((sumV / pixelCount) / maxTheoretical * 600));
        const intensityH = Math.min(100, Math.round((sumH / pixelCount) / maxTheoretical * 600));
        const intensityC = Math.min(100, Math.round((sumC / pixelCount) / (maxTheoretical * 1.414) * 600));
        
        // Draw edges with vibrant color maps — boosted brightness
        const drawEdges = (edges, max, canvas, rFactor, gFactor, bFactor) => {
            const ctx = canvas.getContext('2d');
            const outData = ctx.createImageData(W, H);
            for (let i = 0; i < W * H; i++) {
                const raw = (edges[i] / (max || 1));
                // Aggressive gamma for much brighter edges
                const v = Math.pow(raw, 0.45);
                const base = 6;
                outData.data[i*4]   = Math.min(255, base + Math.floor(v * 249 * rFactor));
                outData.data[i*4+1] = Math.min(255, base + Math.floor(v * 249 * gFactor));
                outData.data[i*4+2] = Math.min(255, base + Math.floor(v * 249 * bFactor));
                outData.data[i*4+3] = 255;
            }
            ctx.putImageData(outData, 0, 0);
        };
        
        // Render all edges to canvases (hidden behind scan overlay)
        drawEdges(vEdges, maxV, vCanvas, 0.1, 0.85, 1.0);
        drawEdges(hEdges, maxH, hCanvas, 1.0, 0.15, 0.9);
        drawEdges(cEdges, maxC, cCanvas, 1.0, 0.85, 0.1);
        
        // === STAGGERED ANIMATED REVEAL ===
        const items = ['original', 'vertical', 'horizontal', 'combined'];
        const delays = [0, 300, 700, 1100];
        const scanDelays = [100, 500, 900, 1300];
        const glowDelays = [500, 1200, 1600, 2000];
        const intensities = { vertical: intensityV, horizontal: intensityH, combined: intensityC };
        
        items.forEach((key, i) => {
            const item = document.getElementById('conv-item-' + key);
            const overlay = document.getElementById('scan-overlay-' + key);
            
            // Step 1: Reveal the card with fade-in
            setTimeout(() => {
                item.classList.add('conv-revealed');
            }, delays[i]);
            
            // Step 2: Start scanning animation
            setTimeout(() => {
                if (overlay) {
                    overlay.classList.add('scanning');
                    overlay.addEventListener('animationend', () => {
                        overlay.classList.remove('scanning');
                        overlay.classList.add('done');
                    }, { once: true });
                }
            }, scanDelays[i]);
            
            // Step 3: Activate glow and intensity bar
            if (key !== 'original') {
                setTimeout(() => {
                    const wrapper = item.querySelector('.conv-canvas-wrapper');
                    if (wrapper) wrapper.classList.add('glow-active');
                    
                    const fill = document.getElementById('intensity-' + key);
                    const val = document.getElementById('intensity-val-' + key);
                    if (fill) fill.style.width = intensities[key] + '%';
                    if (val) val.textContent = intensities[key] + '%';
                }, glowDelays[i]);
            }
        });
        
        // Step 4: Show completion summary and trigger cascade
        const summary = document.getElementById('conv-summary');
        setTimeout(async () => {
            if (summary) summary.style.display = 'flex';
            
            // Run inference and WAIT for it to complete before cascade
            await runRealInference();

            // Cascade to Step 3: ReLU Activation
            setTimeout(runReluLayer, 1000);
        }, 2800);
    }

    // ===== CONVOLUTION SCANNER ENGINE =====
    const scannerFilters = {
        vertical: { name: 'Sobel-X', kernel: [[-1,0,1],[-2,0,2],[-1,0,1]], color: {r:6,g:182,b:212} },
        horizontal: { name: 'Sobel-Y', kernel: [[-1,-2,-1],[0,0,0],[1,2,1]], color: {r:217,g:70,b:239} }
    };

    const SC = {
        playing: false, x: 0, y: 0,
        gray: null, output: null, computed: null,
        W: 16, outW: 14, filter: 'vertical',
        timer: null, speed: 5, img: null
    };

    function initScanner(img) {
        SC.img = img;
        const off = document.createElement('canvas');
        off.width = SC.W; off.height = SC.W;
        const oc = off.getContext('2d');
        oc.drawImage(img, 0, 0, SC.W, SC.W);
        const d = oc.getImageData(0, 0, SC.W, SC.W).data;

        SC.gray = new Float32Array(SC.W * SC.W);
        for (let i = 0; i < SC.W * SC.W; i++) {
            SC.gray[i] = Math.round(d[i*4]*0.299 + d[i*4+1]*0.587 + d[i*4+2]*0.114);
        }
        scannerReset();
    }

    function scannerReset() {
        SC.x = 0; SC.y = 0; SC.playing = false;
        SC.output = new Float32Array(SC.outW * SC.outW);
        SC.computed = new Uint8Array(SC.outW * SC.outW);
        if (SC.timer) { clearTimeout(SC.timer); SC.timer = null; }

        const pb = document.getElementById('scanner-play');
        if (pb) pb.textContent = '▶ Play';

        drawScannerInput();
        drawScannerMath();
        drawScannerOutput();
        updateScannerProgress();
    }

    function drawScannerInput() {
        const cv = document.getElementById('scanner-input-canvas');
        if (!cv || !SC.gray) return;
        const ctx = cv.getContext('2d');
        const cs = cv.width / SC.W; // cell size = 16

        ctx.clearRect(0, 0, cv.width, cv.height);

        // Draw pixel cells
        for (let y = 0; y < SC.W; y++) {
            for (let x = 0; x < SC.W; x++) {
                const v = SC.gray[y * SC.W + x];
                ctx.fillStyle = `rgb(${v},${v},${v})`;
                ctx.fillRect(x*cs, y*cs, cs, cs);
            }
        }

        // Grid lines
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= SC.W; i++) {
            ctx.beginPath(); ctx.moveTo(i*cs, 0); ctx.lineTo(i*cs, cv.height); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, i*cs); ctx.lineTo(cv.width, i*cs); ctx.stroke();
        }

        // Kernel overlay with glow
        const f = scannerFilters[SC.filter];
        const kx = SC.x * cs, ky = SC.y * cs, ks = 3 * cs;

        ctx.save();
        ctx.shadowColor = `rgba(${f.color.r},${f.color.g},${f.color.b},0.9)`;
        ctx.shadowBlur = 16;
        ctx.strokeStyle = `rgba(${f.color.r},${f.color.g},${f.color.b},0.95)`;
        ctx.lineWidth = 3;
        ctx.strokeRect(kx+1, ky+1, ks-2, ks-2);
        ctx.restore();

        // Tint kernel region
        ctx.fillStyle = `rgba(${f.color.r},${f.color.g},${f.color.b},0.12)`;
        ctx.fillRect(kx, ky, ks, ks);

        document.getElementById('scanner-pos').textContent = `Kernel at (${SC.x}, ${SC.y})`;
    }

    function drawScannerMath() {
        const patchEl = document.getElementById('scanner-patch');
        const filterEl = document.getElementById('scanner-filter-grid');
        const resultEl = document.getElementById('scanner-result');
        if (!patchEl || !SC.gray) return;

        patchEl.innerHTML = '';
        filterEl.innerHTML = '';

        const f = scannerFilters[SC.filter];
        let sum = 0;

        for (let ky = 0; ky < 3; ky++) {
            for (let kx = 0; kx < 3; kx++) {
                const px = SC.x + kx, py = SC.y + ky;
                const pv = SC.gray[py * SC.W + px];
                const fv = f.kernel[ky][kx];
                sum += pv * fv;

                // Patch cell
                const pc = document.createElement('div');
                pc.className = 'scanner-cell';
                pc.textContent = Math.round(pv);
                const bright = pv / 255;
                pc.style.background = `rgba(255,255,255,${bright * 0.8 + 0.05})`;
                pc.style.color = pv > 128 ? '#111' : '#eee';
                patchEl.appendChild(pc);

                // Filter cell
                const fc = document.createElement('div');
                fc.className = 'scanner-cell';
                fc.textContent = fv;
                if (fv > 0) fc.style.background = `rgba(34,197,94,${0.15 + fv * 0.2})`;
                else if (fv < 0) fc.style.background = `rgba(239,68,68,${0.15 + Math.abs(fv) * 0.2})`;
                else fc.style.background = 'rgba(255,255,255,0.05)';
                fc.style.color = '#fff';
                filterEl.appendChild(fc);
            }
        }

        const absSum = Math.abs(Math.round(sum));
        resultEl.textContent = Math.round(sum);

        // Color the result based on magnitude
        const intensity = Math.min(absSum / 600, 1);
        resultEl.style.borderColor = `rgba(${f.color.r},${f.color.g},${f.color.b},${0.3 + intensity * 0.7})`;
        resultEl.style.background = `rgba(${f.color.r},${f.color.g},${f.color.b},${intensity * 0.2})`;
        resultEl.style.color = `rgba(${f.color.r},${f.color.g},${f.color.b},${0.6 + intensity * 0.4})`;
        resultEl.style.textShadow = `0 0 ${8 + intensity * 16}px rgba(${f.color.r},${f.color.g},${f.color.b},${intensity * 0.7})`;

        // Store in output
        const oi = SC.y * SC.outW + SC.x;
        SC.output[oi] = Math.abs(sum);
        SC.computed[oi] = 1;
    }

    function drawScannerOutput() {
        const cv = document.getElementById('scanner-output-canvas');
        if (!cv) return;
        const ctx = cv.getContext('2d');
        const cs = cv.width / SC.outW;
        const f = scannerFilters[SC.filter];

        // Find max for normalization
        let mx = 1;
        for (let i = 0; i < SC.outW * SC.outW; i++) {
            if (SC.output[i] > mx) mx = SC.output[i];
        }

        ctx.clearRect(0, 0, cv.width, cv.height);

        // Fill background
        ctx.fillStyle = '#050508';
        ctx.fillRect(0, 0, cv.width, cv.height);

        for (let y = 0; y < SC.outW; y++) {
            for (let x = 0; x < SC.outW; x++) {
                const idx = y * SC.outW + x;
                if (SC.computed[idx]) {
                    const v = Math.pow(SC.output[idx] / mx, 0.5);
                    ctx.fillStyle = `rgba(${Math.floor(f.color.r * v + 6)},${Math.floor(f.color.g * v + 6)},${Math.floor(f.color.b * v + 6)},1)`;
                    ctx.fillRect(x*cs, y*cs, cs, cs);
                }
            }
        }

        // Grid
        ctx.strokeStyle = 'rgba(255,255,255,0.04)';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= SC.outW; i++) {
            ctx.beginPath(); ctx.moveTo(i*cs, 0); ctx.lineTo(i*cs, cv.height); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, i*cs); ctx.lineTo(cv.width, i*cs); ctx.stroke();
        }

        // Highlight current output cell
        if (SC.computed[SC.y * SC.outW + SC.x]) {
            ctx.save();
            ctx.shadowColor = `rgba(${f.color.r},${f.color.g},${f.color.b},0.8)`;
            ctx.shadowBlur = 10;
            ctx.strokeStyle = `rgba(255,255,255,0.7)`;
            ctx.lineWidth = 2;
            ctx.strokeRect(SC.x*cs+1, SC.y*cs+1, cs-2, cs-2);
            ctx.restore();
        }
    }

    function updateScannerProgress() {
        const total = SC.outW * SC.outW;
        let done = 0;
        for (let i = 0; i < total; i++) { if (SC.computed[i]) done++; }
        const pf = document.getElementById('scanner-progress-fill');
        const pt = document.getElementById('scanner-progress-text');
        if (pf) pf.style.width = (done / total * 100) + '%';
        if (pt) pt.textContent = `${done} / ${total}`;
    }

    function scannerAdvance() {
        SC.x++;
        if (SC.x >= SC.outW) { SC.x = 0; SC.y++; }
        if (SC.y >= SC.outW) return false;
        return true;
    }

    function scannerDoStep() {
        if (!SC.gray) return;
        drawScannerMath();
        drawScannerInput();
        drawScannerOutput();
        updateScannerProgress();
    }

    function scannerPlay() {
        SC.playing = true;
        const pb = document.getElementById('scanner-play');
        if (pb) pb.textContent = '⏸ Pause';

        function tick() {
            if (!SC.playing) return;
            scannerDoStep();
            if (!scannerAdvance()) {
                SC.playing = false;
                if (pb) pb.textContent = '✓ Done';
                return;
            }
            const delay = Math.max(10, 220 - SC.speed * 22);
            SC.timer = setTimeout(tick, delay);
        }
        tick();
    }

    function scannerPause() {
        SC.playing = false;
        if (SC.timer) { clearTimeout(SC.timer); SC.timer = null; }
        const pb = document.getElementById('scanner-play');
        if (pb) pb.textContent = '▶ Play';
    }

    // Wire up scanner controls
    const scanPlayBtn = document.getElementById('scanner-play');
    const scanStepBtn = document.getElementById('scanner-step');
    const scanResetBtn = document.getElementById('scanner-reset');
    const scanSpeedInput = document.getElementById('scanner-speed');

    if (scanPlayBtn) {
        scanPlayBtn.addEventListener('click', () => {
            if (SC.playing) scannerPause();
            else scannerPlay();
        });
    }
    if (scanStepBtn) {
        scanStepBtn.addEventListener('click', () => {
            scannerPause();
            scannerDoStep();
            scannerAdvance();
        });
    }
    if (scanResetBtn) {
        scanResetBtn.addEventListener('click', () => {
            scannerReset();
            if (SC.img) initScanner(SC.img);
        });
    }
    if (scanSpeedInput) {
        scanSpeedInput.addEventListener('input', (e) => { SC.speed = parseInt(e.target.value); });
    }

    // Filter tab switching
    document.querySelectorAll('.scanner-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.scanner-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            SC.filter = tab.getAttribute('data-filter');
            if (SC.img) {
                scannerReset();
                initScanner(SC.img);
            }
        });
    });
    
    // ===== NEW FULL-STEP PIPELINE LOGIC (ReLU -> BN -> Pool) =====
    
    // Shared state to pass image data between layers
    const LAYER_DATA = {
        reluData: null,
        bnData: null,
        predictions: null   // Will hold real softmax probabilities after inference
    };

    // ===== CLEAN MNIST DATA LOADER (no UI dependencies) =====
    let _mnistDataClean = null;
    async function loadMnistDataClean(statusCb) {
        if (_mnistDataClean) return _mnistDataClean;

        const imgUrl = 'models/mnist_images.png';
        const labelsUrl = 'models/mnist_labels_uint8';

        if (statusCb) statusCb('⏳ Fetching MNIST labels...');
        const labelsRes = await fetch(labelsUrl);
        if (!labelsRes.ok) throw new Error(`Labels fetch failed: ${labelsRes.status}`);
        const labelsBuffer = await labelsRes.arrayBuffer();
        const rawLabels = new Uint8Array(labelsBuffer); // Exactly 65000 * 10 one-hot, no header

        if (statusCb) statusCb('⏳ Fetching MNIST images...');
        const img = await new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error('MNIST image failed to load'));
            img.src = imgUrl;
        });

        // Use 5,000 samples for fast, reliable accuracy (~95%)
        const NUM_TRAIN = 5000;
        
        const c = document.createElement('canvas');
        c.width = img.width; c.height = img.height;
        const ctx = c.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        const imgData = ctx.getImageData(0, 0, c.width, c.height);
        
        const float32 = new Float32Array(NUM_TRAIN * 784);
        for (let i = 0; i < NUM_TRAIN * 784; i++) {
            float32[i] = imgData.data[i * 4] / 255.0;
        }

        const xs = tf.tensor4d(float32, [NUM_TRAIN, 28, 28, 1]);
        
        // rawLabels is already one-hot encoded (10 bytes per label)
        const ysArr = new Float32Array(NUM_TRAIN * 10);
        for(let i=0; i<NUM_TRAIN * 10; i++) ysArr[i] = rawLabels[i];
        const ys = tf.tensor2d(ysArr, [NUM_TRAIN, 10]);

        _mnistDataClean = { xs, ys };
        return _mnistDataClean;
    }

    // ===== REAL INFERENCE USING TRAINED MNIST MODEL =====
    async function runRealInference() {
        if (!model || !currentUploadedImage) return;

        try {
            // Preprocess: draw to 28x28 grayscale
            const off = document.createElement('canvas');
            off.width = 28; off.height = 28;
            const octx = off.getContext('2d');
            // White background (MNIST has white digits on black, but user images may vary)
            octx.fillStyle = 'black';
            octx.fillRect(0, 0, 28, 28);
            octx.drawImage(currentUploadedImage, 0, 0, 28, 28);
            const pixels = octx.getImageData(0, 0, 28, 28).data;

            const float32 = new Float32Array(28 * 28);
            for (let i = 0; i < 28 * 28; i++) {
                float32[i] = (pixels[i*4]*0.299 + pixels[i*4+1]*0.587 + pixels[i*4+2]*0.114) / 255;
            }

            const tensor = tf.tensor4d(float32, [1, 28, 28, 1]);
            const outputTensor = model.predict(tensor);
            const probs = Array.from(outputTensor.dataSync());
            tf.dispose([tensor, outputTensor]);

            LAYER_DATA.predictions = probs; // float array [10]
            console.log('Real digit predictions:', probs);
        } catch (e) {
            console.error('Inference error:', e);
        }
    }

    function runReluLayer() {
        const step2_3 = document.getElementById('step-connector-2-3');
        const step3 = document.getElementById('step-3');
        
        step2_3.style.display = 'flex';
        setTimeout(() => {
            step3.style.display = 'block';
            setTimeout(() => step3.classList.add('active-step'), 50);
            
            // Copy input from Combined Conv
            const inCanvas = document.getElementById('relu-input-canvas');
            const outCanvas = document.getElementById('relu-output-canvas');
            const srcCanvas = document.getElementById('conv-canvas-combined');
            
            if(!srcCanvas || !inCanvas || !outCanvas) return;
            
            const srcCtx = srcCanvas.getContext('2d', {willReadFrequently: true});
            const inCtx = inCanvas.getContext('2d');
            const outCtx = outCanvas.getContext('2d');
            
            // 128x128 canvases
            const imgData = srcCtx.getImageData(0, 0, 128, 128);
            inCtx.putImageData(imgData, 0, 0);
            
            // Process ReLU
            const reluData = outCtx.createImageData(128, 128);
            reluData.data.set(imgData.data);
            const rd = reluData.data;
            for (let i = 0; i < rd.length; i += 4) {
                const brightness = (rd[i] + rd[i+1] + rd[i+2]) / 3;
                if (brightness < 45) { // Threshold
                    rd[i] = 0; rd[i+1] = 0; rd[i+2] = 0;
                } else {
                    rd[i] = Math.min(255, rd[i] * 1.2);
                    rd[i+1] = Math.min(255, rd[i+1] * 1.2);
                    rd[i+2] = Math.min(255, rd[i+2] * 1.2);
                }
            }
            LAYER_DATA.reluData = reluData;
            
            // Animation: Sweeping laser
            const overlay = document.getElementById('scan-overlay-relu');
            if (overlay) overlay.style.display = 'none'; // Hide CSS overlay
            
            const animData = outCtx.createImageData(128, 128);
            animData.data.set(imgData.data);
            let sweepY = 0;
            let hasCascadedReLU = false;
            
            function stepReluSweep() {
                // Copy new rows from reluData to animData
                const startY = Math.floor(sweepY);
                sweepY += 1.5; // Slow down to 1.5px per frame (approx 1.5 seconds total)
                const endY = Math.floor(sweepY);
                
                if (endY > startY) {
                    const startIdx = startY * 128 * 4;
                    const endIdx = Math.min(endY * 128 * 4, animData.data.length);
                    for (let i = startIdx; i < endIdx; i++) {
                        animData.data[i] = reluData.data[i];
                    }
                    outCtx.putImageData(animData, 0, 0);
                } else {
                    outCtx.putImageData(animData, 0, 0); // Need to clear previous stroke
                }
                
                // Draw laser line
                outCtx.save();
                outCtx.beginPath();
                outCtx.moveTo(0, sweepY);
                outCtx.lineTo(128, sweepY);
                outCtx.strokeStyle = 'rgba(6, 182, 212, 1)'; // Cyan
                outCtx.shadowColor = 'rgba(6, 182, 212, 1)';
                outCtx.shadowBlur = 12;
                outCtx.lineWidth = 2;
                outCtx.stroke();
                outCtx.restore();
                
                if (sweepY >= 128) {
                    outCtx.putImageData(reluData, 0, 0); // final clean draw
                    
                    if (!hasCascadedReLU) {
                        hasCascadedReLU = true;
                        setTimeout(runBatchNormLayer, 800); // Cascade
                    }
                    
                    // Reset and loop continuously
                    setTimeout(() => {
                        sweepY = 0;
                        requestAnimationFrame(stepReluSweep);
                    }, 2000);
                    return;
                }
                requestAnimationFrame(stepReluSweep);
            }
            
            setTimeout(stepReluSweep, 1000); // Wait 1 full second for the step to fade in before animating
            
        }, 1200);
    }

    function runBatchNormLayer() {
        const step3_4 = document.getElementById('step-connector-3-4');
        const step4 = document.getElementById('step-4');
        
        step3_4.style.display = 'flex';
        setTimeout(() => {
            step4.style.display = 'block';
            setTimeout(() => step4.classList.add('active-step'), 50);
            
            const inCanvas = document.getElementById('bn-input-canvas');
            const outCanvas = document.getElementById('bn-output-canvas');
            if(!inCanvas || !outCanvas || !LAYER_DATA.reluData) return;
            
            inCanvas.getContext('2d').putImageData(LAYER_DATA.reluData, 0, 0);
            const outCtx = outCanvas.getContext('2d');
            
            // Process BN
            const bnData = outCtx.createImageData(128, 128);
            bnData.data.set(LAYER_DATA.reluData.data);
            const bd = bnData.data;
            for (let i = 0; i < bd.length; i += 4) {
                if (bd[i] > 0 || bd[i+1] > 0 || bd[i+2] > 0) {
                    const r = bd[i], g = bd[i+1], b = bd[i+2];
                    bd[i] = Math.floor((r * 0.3 + b * 0.7)); // shift colors
                    bd[i+1] = Math.floor((g * 0.8 + 20));
                    bd[i+2] = Math.floor((b * 0.2 + r * 0.8 + 40));
                }
            }
            LAYER_DATA.bnData = bnData;
            
            // Animation: Radial stabilizer
            const overlay = document.getElementById('scan-overlay-bn');
            if (overlay) overlay.style.display = 'none'; // Hide CSS overlay
            
            const animData = outCtx.createImageData(128, 128);
            animData.data.set(LAYER_DATA.reluData.data);
            let radius = 0;
            let hasCascadedBN = false;
            const maxRadius = 128 * Math.SQRT2; // reach corners
            const cx = 64, cy = 64;
            
            function stepBNRadial() {
                // Update pixels inside radius
                const rSq = radius * radius;
                for (let y = 0; y < 128; y++) {
                    const dy = y - cy;
                    const dySq = dy * dy;
                    for (let x = 0; x < 128; x++) {
                        const dx = x - cx;
                        if (dx * dx + dySq <= rSq) {
                            const idx = (y * 128 + x) * 4;
                            animData.data[idx] = bnData.data[idx];
                            animData.data[idx+1] = bnData.data[idx+1];
                            animData.data[idx+2] = bnData.data[idx+2];
                        }
                    }
                }
                
                outCtx.putImageData(animData, 0, 0);
                
                // Draw expanding ring
                outCtx.save();
                outCtx.beginPath();
                outCtx.arc(cx, cy, radius, 0, Math.PI * 2);
                outCtx.strokeStyle = 'rgba(217, 70, 239, 1)'; // Magenta
                outCtx.shadowColor = 'rgba(217, 70, 239, 1)';
                outCtx.shadowBlur = 12;
                outCtx.lineWidth = 2;
                outCtx.stroke();
                outCtx.restore();
                
                radius += 1.5; // Slow down to 1.5px per frame (approx 2 seconds total)
                
                if (radius >= maxRadius) {
                    outCtx.putImageData(bnData, 0, 0); // final clean draw
                    
                    if (!hasCascadedBN) {
                        hasCascadedBN = true;
                        setTimeout(runMaxPoolLayer, 800); // Cascade
                    }
                    
                    // Reset and loop continuously
                    setTimeout(() => {
                        radius = 0;
                        requestAnimationFrame(stepBNRadial);
                    }, 2000);
                    return;
                }
                requestAnimationFrame(stepBNRadial);
            }
            
            setTimeout(stepBNRadial, 1000); // Wait 1 full second for the step to fade in before animating
            
        }, 1200);
    }

    // ===== Max Pooling State Machine =====
    const POOL = {
        playing: false, timer: null, speed: 5,
        px: 0, py: 0, outW: 64, W: 128,
        data: null, // BN output array
        stepState: 0, // 0: select, 1: highlight max, 2: write out
        chosenColor: null
    };

    function runMaxPoolLayer() {
        const step4_5 = document.getElementById('step-connector-4-5');
        const step5 = document.getElementById('step-5');
        
        step4_5.style.display = 'flex';
        setTimeout(() => {
            step5.style.display = 'block';
            setTimeout(() => step5.classList.add('active-step'), 50);
            
            const inCanvas = document.getElementById('pool-input-canvas');
            const outCanvas = document.getElementById('pool-output-canvas');
            if(!inCanvas || !outCanvas || !LAYER_DATA.bnData) return;
            
            inCanvas.getContext('2d').putImageData(LAYER_DATA.bnData, 0, 0);
            const outCtx = outCanvas.getContext('2d');
            outCtx.clearRect(0, 0, 64, 64);
            outCtx.fillStyle = '#050508';
            outCtx.fillRect(0, 0, 64, 64);
            
            POOL.data = LAYER_DATA.bnData.data;
            POOL.px = 0; POOL.py = 0;
            POOL.stepState = 0;
            POOL.playing = true;
            document.getElementById('pool-play').textContent = '⏸ Pause Downsampling';
            
            document.getElementById('pool-scanner-box').style.display = 'block';
            
            // Start the pooling machine
            setTimeout(tickPoolScanner, 600);
            
        }, 1200);
    }

    function drawPoolScannerBox(srcX, srcY) {
        const scanner = document.getElementById('pool-scanner-box');
        if (!scanner) return;
        scanner.style.width = (2 / 128 * 100) + '%';
        scanner.style.height = (2 / 128 * 100) + '%';
        scanner.style.left = (srcX / 128 * 100) + '%';
        scanner.style.top = (srcY / 128 * 100) + '%';
    }

    function tickPoolScanner() {
        if (!POOL.playing) return;
        
        const delay = doPoolStep();
        if (delay === -1) {
            POOL.playing = false;
            const playBtn = document.getElementById('pool-play');
            if (playBtn) playBtn.textContent = '✓ Complete';
            const scanner = document.getElementById('pool-scanner-box');
            if (scanner) scanner.style.display = 'none';
            document.getElementById('pool-status').textContent = 'Downsampling complete. Output size 64x64.';
            
            // Trigger cascade to step 6 (Residual)
            setTimeout(runResidualLayer, 1000);
            return;
        }
        
        POOL.timer = setTimeout(tickPoolScanner, delay);
    }
    
    function fastPoolStep() {
        if (POOL.py >= POOL.outW) return false;
        
        const srcX = POOL.px * 2;
        const srcY = POOL.py * 2;
        let maxR = 0, maxG = 0, maxB = 0;
        
        for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) {
                const idx = ((srcY + dy) * POOL.W + (srcX + dx)) * 4;
                if (POOL.data[idx] > maxR) maxR = POOL.data[idx];
                if (POOL.data[idx+1] > maxG) maxG = POOL.data[idx+1];
                if (POOL.data[idx+2] > maxB) maxB = POOL.data[idx+2];
            }
        }
        
        const outCtx = document.getElementById('pool-output-canvas').getContext('2d');
        outCtx.fillStyle = `rgb(${maxR},${maxG},${maxB})`;
        outCtx.fillRect(POOL.px, POOL.py, 1, 1);
        
        POOL.px++;
        if (POOL.px >= POOL.outW) {
            POOL.px = 0;
            POOL.py++;
        }
        return true;
    }

    function doPoolStep() {
        if (POOL.py >= POOL.outW) return -1;
        
        if (POOL.speed === 10) {
            // Fast mode: skip UI updates, blast through pixels
            for(let i=0; i<80; i++) {
                if (!fastPoolStep()) return -1;
            }
            return 10;
        }
        
        const srcX = POOL.px * 2;
        const srcY = POOL.py * 2;
        const cells = [
            document.getElementById('pool-cell-0'),
            document.getElementById('pool-cell-1'),
            document.getElementById('pool-cell-2'),
            document.getElementById('pool-cell-3')
        ];
        
        // State 0: Select region
        if (POOL.stepState === 0) {
            drawPoolScannerBox(srcX, srcY);
            
            cells.forEach(c => {
                c.className = 'pool-cell';
                c.textContent = '-';
                c.style.background = 'rgba(255,255,255,0.05)';
            });
            document.getElementById('pool-status').textContent = `Scanning region (${srcX}, ${srcY})`;
            
            POOL.stepState = 1;
            return Math.max(10, 300 - POOL.speed * 28);
        }
        
        let vals = [];
        let rawColors = [];
        for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) {
                const idx = ((srcY + dy) * POOL.W + (srcX + dx)) * 4;
                const r = POOL.data[idx], g = POOL.data[idx+1], b = POOL.data[idx+2];
                const val = Math.round((r + g + b) / 3);
                vals.push(val);
                rawColors.push({r,g,b});
            }
        }
        
        // State 1: Highlight max
        if (POOL.stepState === 1) {
            let maxVal = -1;
            let maxIdx = -1;
            vals.forEach((v, i) => {
                cells[i].textContent = v;
                if (v > maxVal) { maxVal = v; maxIdx = i; }
            });
            
            cells.forEach((c, i) => {
                if (i === maxIdx) {
                    c.classList.add('active-max');
                    document.getElementById('pool-status').textContent = `Max value found: ${maxVal}`;
                } else {
                    c.classList.add('fade-out');
                }
            });
            
            let maxR = Math.max(rawColors[0].r, rawColors[1].r, rawColors[2].r, rawColors[3].r);
            let maxG = Math.max(rawColors[0].g, rawColors[1].g, rawColors[2].g, rawColors[3].g);
            let maxB = Math.max(rawColors[0].b, rawColors[1].b, rawColors[2].b, rawColors[3].b);
            POOL.chosenColor = {r: maxR, g: maxG, b: maxB};
            
            POOL.stepState = 2;
            return Math.max(20, 600 - POOL.speed * 55);
        }
        
        // State 2: Write out
        if (POOL.stepState === 2) {
            const outCtx = document.getElementById('pool-output-canvas').getContext('2d');
            const c = POOL.chosenColor;
            outCtx.fillStyle = `rgb(${c.r},${c.g},${c.b})`;
            outCtx.fillRect(POOL.px, POOL.py, 1, 1);
            
            POOL.stepState = 0;
            POOL.px++;
            if (POOL.px >= POOL.outW) {
                POOL.px = 0;
                POOL.py++;
            }
            
            return Math.max(5, 50 - POOL.speed * 4);
        }
    }

    // Step 5 Event Listeners
    document.getElementById('pool-play')?.addEventListener('click', () => {
        if (POOL.playing) {
            POOL.playing = false;
            clearTimeout(POOL.timer);
            document.getElementById('pool-play').textContent = '▶ Play Downsampling';
        } else {
            if (POOL.py >= POOL.outW) return; // already done
            POOL.playing = true;
            document.getElementById('pool-play').textContent = '⏸ Pause Downsampling';
            tickPoolScanner();
        }
    });

    document.getElementById('pool-step')?.addEventListener('click', () => {
        POOL.playing = false;
        clearTimeout(POOL.timer);
        document.getElementById('pool-play').textContent = '▶ Play Downsampling';
        doPoolStep();
    });

    document.getElementById('pool-speed')?.addEventListener('input', (e) => {
        POOL.speed = parseInt(e.target.value);
    });

    // Hover interactivity on output map to show input region
    const poolOutputCanvas = document.getElementById('pool-output-canvas');
    if (poolOutputCanvas) {
        poolOutputCanvas.addEventListener('mousemove', (e) => {
            // Only allow hover inspection if animation is complete or paused
            if (POOL.playing) return;
            
            const rect = poolOutputCanvas.getBoundingClientRect();
            // Calculate which 1x1 pixel in the 64x64 canvas was hovered
            // The canvas is rendered at a max-width of 250px, but native width is 64.
            const scaleX = 64 / rect.width;
            const scaleY = 64 / rect.height;
            const x = Math.floor((e.clientX - rect.left) * scaleX);
            const y = Math.floor((e.clientY - rect.top) * scaleY);
            
            if (x >= 0 && x < 64 && y >= 0 && y < 64) {
                // Show scanner box at corresponding 2x2 input region
                const scanner = document.getElementById('pool-scanner-box');
                scanner.style.display = 'block';
                drawPoolScannerBox(x * 2, y * 2);
                
                // Show hover highlight box on output
                const hBox = document.getElementById('pool-hover-box');
                hBox.style.display = 'block';
                hBox.style.width = (1 / 64 * 100) + '%';
                hBox.style.height = (1 / 64 * 100) + '%';
                hBox.style.left = (x / 64 * 100) + '%';
                hBox.style.top = (y / 64 * 100) + '%';
            }
        });
        
        poolOutputCanvas.addEventListener('mouseleave', () => {
            // Only hide if we are paused/done. If playing, the tick logic handles scanner.
            if (!POOL.playing) {
                document.getElementById('pool-scanner-box').style.display = 'none';
            }
            document.getElementById('pool-hover-box').style.display = 'none';
        });
    }

    // ===== RESIDUAL BLOCK (STEP 6) =====
    function runResidualLayer() {
        const step5_6 = document.getElementById('step-connector-5-6');
        const step6 = document.getElementById('step-6');
        
        step5_6.style.display = 'flex';
        setTimeout(() => {
            step6.style.display = 'block';
            setTimeout(() => step6.classList.add('active-step'), 50);
            
            // Get canvases
            const inputCtx = document.getElementById('res-input-canvas').getContext('2d');
            const mainCtx = document.getElementById('res-main-canvas').getContext('2d');
            const skipCtx = document.getElementById('res-skip-canvas').getContext('2d');
            const outCtx = document.getElementById('res-output-canvas').getContext('2d');
            
            if(!inputCtx || !mainCtx || !skipCtx || !outCtx) return;
            
            // 1. Copy Pool Data to Input
            const poolCanvas = document.getElementById('pool-output-canvas');
            const poolCtx = poolCanvas.getContext('2d');
            const poolImgData = poolCtx.getImageData(0, 0, 64, 64);
            
            inputCtx.putImageData(poolImgData, 0, 0);
            
            // 2. Animate Skip Path (just draw it, it's an identity connection)
            skipCtx.putImageData(poolImgData, 0, 0);
            
            // 3. Animate Main Path transformation (simulate Conv+BN+ReLU)
            const mainData = mainCtx.createImageData(64, 64);
            mainData.data.set(poolImgData.data);
            const md = mainData.data;
            
            let frame = 0;
            const totalFrames = 60; // 1 second transformation
            let hasCascadedResidual = false;
            
            function animateMainPath() {
                frame++;
                const progress = frame / totalFrames; // 0 to 1
                
                // Shift colors slightly towards purple/blue to show feature transformation
                for(let i = 0; i < md.length; i+=4) {
                    if (md[i] > 10 || md[i+1] > 10 || md[i+2] > 10) {
                        const origR = poolImgData.data[i];
                        const origG = poolImgData.data[i+1];
                        const origB = poolImgData.data[i+2];
                        
                        // Target color shift (purple/blue feature extraction)
                        const targetR = Math.min(255, origR * 0.8 + 50);
                        const targetG = Math.max(0, origG * 0.5);
                        const targetB = Math.min(255, origB * 1.2 + 80);
                        
                        md[i] = origR + (targetR - origR) * progress;
                        md[i+1] = origG + (targetG - origG) * progress;
                        md[i+2] = origB + (targetB - origB) * progress;
                    }
                }
                mainCtx.putImageData(mainData, 0, 0);
                
                if (frame < totalFrames) {
                    requestAnimationFrame(animateMainPath);
                } else {
                    // Start merge animation
                    setTimeout(animateMerge, 500);
                }
            }
            
            function animateMerge() {
                // Merge operation: F(x) + x
                const statusEl = document.getElementById('res-status');
                if (statusEl) statusEl.textContent = 'F(x) + x : Addition complete!';
                
                const mergedData = outCtx.createImageData(64, 64);
                
                for(let i = 0; i < mergedData.data.length; i+=4) {
                    const r = mainData.data[i] + poolImgData.data[i];
                    const g = mainData.data[i+1] + poolImgData.data[i+1];
                    const b = mainData.data[i+2] + poolImgData.data[i+2];
                    
                    mergedData.data[i] = Math.min(255, r);
                    mergedData.data[i+1] = Math.min(255, g);
                    mergedData.data[i+2] = Math.min(255, b);
                    mergedData.data[i+3] = 255; // Full alpha
                }
                outCtx.putImageData(mergedData, 0, 0);
                
                // Glow effect on output canvas to signify the burst of combined features
                const outCanvasEl = document.getElementById('res-output-canvas');
                outCanvasEl.style.boxShadow = '0 0 30px rgba(255, 255, 255, 0.8)';
                setTimeout(() => outCanvasEl.style.boxShadow = '', 800);
                
                if (!hasCascadedResidual) {
                    hasCascadedResidual = true;
                    // Trigger cascade to Flatten layer (now Step 7)
                    setTimeout(runFlattenLayer, 1500);
                }
                
                // Continuous loop
                setTimeout(() => {
                    frame = 0;
                    if (statusEl) statusEl.textContent = 'Combining original features with learned transformations...';
                    outCtx.clearRect(0, 0, 64, 64);
                    animateMainPath();
                }, 4000);
            }
            
            setTimeout(animateMainPath, 1000);
            
        }, 1200);
    }

    // ===== FLATTEN LAYER (STEP 7) =====
    function runFlattenLayer() {
        const step6_7 = document.getElementById('step-connector-6-7');
        const step7 = document.getElementById('step-7');
        
        step6_7.style.display = 'flex';
        setTimeout(() => {
            step7.style.display = 'block';
            setTimeout(() => step7.classList.add('active-step'), 50);
            
            const inCanvas = document.getElementById('flatten-input-canvas');
            const outCanvas = document.getElementById('flatten-vector-canvas');
            if (!inCanvas || !outCanvas) return;
            
            // Get the final 64x64 output from the Residual Block
            const resCanvas = document.getElementById('res-output-canvas');
            const resCtx = resCanvas.getContext('2d');
            const poolImgData = resCtx.getImageData(0, 0, 64, 64);
            
            inCanvas.getContext('2d').putImageData(poolImgData, 0, 0);
            const outCtx = outCanvas.getContext('2d');
            outCtx.clearRect(0, 0, 1, 4096);
            
            // Enable button
            const btn = document.getElementById('btn-start-flatten');
            btn.style.display = 'inline-block';
            
            let isFlattening = false;
            btn.onclick = () => {
                if(isFlattening) return;
                isFlattening = true;
                btn.style.display = 'none';
                startFlattenAnimation(poolImgData.data);
            };
            
            // Automatically click the button after a delay
            setTimeout(() => {
                if(!isFlattening) btn.click();
            }, 1500);
            
        }, 1200);
    }

    function startFlattenAnimation(pixelData) {
        document.getElementById('flatten-status').textContent = 'Converting 64x64 matrix to 4096x1 vector...';
        
        const overlay = document.getElementById('flatten-overlay-canvas');
        const ctx = overlay.getContext('2d');
        
        // Resize overlay to perfectly match the step-card body dimensions
        const bodyRect = document.getElementById('flatten-body').getBoundingClientRect();
        overlay.width = bodyRect.width;
        overlay.height = bodyRect.height;
        
        // Get coordinates of the source and destination
        const leftWrap = document.getElementById('flatten-left-wrap').getBoundingClientRect();
        const rightWrap = document.getElementById('flatten-right-wrap').getBoundingClientRect();
        const outCanvas = document.getElementById('flatten-vector-canvas');
        const outCtx = outCanvas.getContext('2d');
        
        // Animate 1 row (64 pixels) at a time
        let currentRow = 0;
        let hasCascadedFlatten = false;
        
        function animateRow() {
            if (currentRow >= 64) {
                document.getElementById('flatten-status').textContent = 'Flattening complete. Vector shape: (4096, 1)';
                
                if (!hasCascadedFlatten) {
                    hasCascadedFlatten = true;
                    setTimeout(runFullyConnectedLayer, 1200);
                }
                
                // Reset and loop continuously
                setTimeout(() => {
                    currentRow = 0;
                    outCtx.clearRect(0, 0, 1, 4096);
                    document.getElementById('flatten-right-wrap').scrollTop = 0;
                    document.getElementById('flatten-status').textContent = 'Converting 64x64 matrix to 4096x1 vector...';
                    animateRow();
                }, 3000);
                return;
            }
            
            const particles = [];
            const rowY = currentRow; // 0 to 63
            
            // Calculate starting visual positions (relative to overlay)
            const leftScale = leftWrap.width / 64;
            const startOffsetX = leftWrap.left - bodyRect.left;
            const startOffsetY = leftWrap.top - bodyRect.top;
            
            // Dest positions (Aim for the center-top of the container)
            const destOffsetX = rightWrap.left - bodyRect.left + (rightWrap.width / 2); 
            const destOffsetY = rightWrap.top - bodyRect.top + 10;
            
            for (let x = 0; x < 64; x++) {
                const idx = (rowY * 64 + x) * 4;
                const r = pixelData[idx];
                const g = pixelData[idx+1];
                const b = pixelData[idx+2];
                // Skip very dark pixels for performance and aesthetics (looks like only active features are flowing)
                if (r < 10 && g < 10 && b < 10) continue; 
                
                particles.push({
                    x: startOffsetX + (x * leftScale),
                    y: startOffsetY + (rowY * leftScale),
                    destX: destOffsetX,
                    destY: destOffsetY,
                    r, g, b,
                    progress: 0,
                    speed: 0.03 + Math.random() * 0.02,
                    ctrlX: startOffsetX + 100 + Math.random() * 100, // curve
                    ctrlY: startOffsetY - 50 + Math.random() * 200
                });
            }
            
            if (particles.length === 0) {
                // Instantly write to output and go next
                writeRowToVector(currentRow, pixelData, outCtx);
                currentRow++;
                requestAnimationFrame(animateRow);
                return;
            }
            
            function renderParticles() {
                ctx.clearRect(0, 0, overlay.width, overlay.height);
                let allDone = true;
                
                particles.forEach(p => {
                    if (p.progress < 1) {
                        p.progress += p.speed;
                        if (p.progress > 1) p.progress = 1;
                        allDone = false;
                        
                        // Quadratic bezier
                        const t = p.progress;
                        const invT = 1 - t;
                        const curX = invT * invT * p.x + 2 * invT * t * p.ctrlX + t * t * p.destX;
                        const curY = invT * invT * p.y + 2 * invT * t * p.ctrlY + t * t * p.destY;
                        
                        ctx.fillStyle = `rgb(${p.r},${p.g},${p.b})`;
                        ctx.beginPath();
                        ctx.arc(curX, curY, 2.5, 0, Math.PI * 2);
                        ctx.fill();
                    }
                });
                
                if (allDone) {
                    ctx.clearRect(0, 0, overlay.width, overlay.height);
                    writeRowToVector(currentRow, pixelData, outCtx);
                    
                    // Auto-scroll the vector container down slightly
                    const container = document.getElementById('flatten-right-wrap');
                    container.scrollTop = (currentRow * 64) - 100;
                    
                    currentRow++;
                    // Delay slightly between rows for a scanning effect
                    if (currentRow % 2 === 0) {
                        setTimeout(() => requestAnimationFrame(animateRow), 10);
                    } else {
                        requestAnimationFrame(animateRow);
                    }
                } else {
                    requestAnimationFrame(renderParticles);
                }
            }
            
            renderParticles();
        }
        
        animateRow();
    }
    
    function writeRowToVector(row, pixelData, outCtx) {
        // Write the 64 pixels into the 1x4096 canvas
        const imgData = outCtx.createImageData(1, 64);
        for (let x = 0; x < 64; x++) {
            const srcIdx = (row * 64 + x) * 4;
            const destIdx = x * 4;
            imgData.data[destIdx] = pixelData[srcIdx];
            imgData.data[destIdx+1] = pixelData[srcIdx+1];
            imgData.data[destIdx+2] = pixelData[srcIdx+2];
            imgData.data[destIdx+3] = 255;
        }
        outCtx.putImageData(imgData, 0, row * 64);
    }

    // ===== FULLY CONNECTED LAYER (STEP 8) =====
    function runFullyConnectedLayer() {
        const conn8 = document.getElementById('step-connector-7-8');
        const step8 = document.getElementById('step-8');
        conn8.style.display = 'flex';
        setTimeout(() => {
            step8.style.display = 'block';
            setTimeout(() => step8.classList.add('active-step'), 50);
            
            const canvas = document.getElementById('fc-canvas');
            const tooltip = document.getElementById('fc-tooltip');
            const W = 700, H = 340;
            canvas.width = W;
            canvas.height = H;
            const ctx = canvas.getContext('2d');

            // Network structure: columns of nodes
            // [Input(6)] -> [Hidden1(8)] -> [Hidden2(8)] -> [Output(10)]
            const LAYERS = [
                { label: 'Input', count: 6,  x: 80  },
                { label: 'Hidden 1', count: 8, x: 240 },
                { label: 'Hidden 2', count: 8, x: 430 },
                { label: 'Output', count: 10, x: 600 },
            ];
            
            // Compute node positions
            const nodes = LAYERS.map(layer => {
                const positions = [];
                const spacing = H / (layer.count + 1);
                for (let i = 0; i < layer.count; i++) {
                    const activation = Math.random(); // simulated
                    positions.push({ x: layer.x, y: spacing * (i + 1), act: activation, label: layer.label });
                }
                return positions;
            });

            // Draw static connections with weight-based opacity
            function drawConnections(alpha) {
                for (let li = 0; li < nodes.length - 1; li++) {
                    for (const nA of nodes[li]) {
                        for (const nB of nodes[li + 1]) {
                            const weight = (nA.act + nB.act) / 2;
                            const opacity = 0.05 + weight * 0.25 * alpha;
                            ctx.beginPath();
                            ctx.moveTo(nA.x, nA.y);
                            ctx.lineTo(nB.x, nB.y);
                            ctx.strokeStyle = weight > 0.6
                                ? `rgba(139, 92, 246, ${opacity})`   // purple for strong
                                : `rgba(100, 116, 139, ${opacity})`; // grey for weak
                            ctx.lineWidth = 0.8;
                            ctx.stroke();
                        }
                    }
                }
            }

            // Draw nodes
            function drawNodes(glowNodes) {
                nodes.forEach((layer, li) => {
                    layer.forEach((n, ni) => {
                        const isGlowing = glowNodes && glowNodes[li] === ni;
                        const radius = 10;
                        ctx.beginPath();
                        ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
                        // Node fill based on activation
                        const brightness = Math.floor(n.act * 200);
                        ctx.fillStyle = isGlowing
                            ? `rgb(200, 180, 255)`
                            : `rgb(${brightness}, ${Math.floor(brightness*0.5)}, ${Math.min(255, brightness + 80)})`;
                        ctx.fill();
                        if (isGlowing || n.act > 0.7) {
                            ctx.shadowColor = 'rgba(139,92,246,0.9)';
                            ctx.shadowBlur = 18;
                            ctx.beginPath();
                            ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
                            ctx.fill();
                            ctx.shadowBlur = 0;
                        }
                        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    });
                });

                // Layer labels
                LAYERS.forEach(l => {
                    ctx.fillStyle = 'rgba(255,255,255,0.35)';
                    ctx.font = '11px Inter, sans-serif';
                    ctx.textAlign = 'center';
                    ctx.fillText(l.label, l.x, H - 10);
                });
            }

            // Animate signal pulses (a glowing dot sweeping layer → layer)
            let pulseX = LAYERS[0].x;
            let pulseTargetIdx = 1;
            let pulseFraction = 0;
            let hasCascadedFC = false;
            let loopCount = 0;

            function fcLoop() {
                ctx.clearRect(0, 0, W, H);
                drawConnections(1);
                drawNodes(null);

                const fromLayer = LAYERS[pulseTargetIdx - 1];
                const toLayer = LAYERS[pulseTargetIdx];
                pulseFraction += 0.025;

                const px = fromLayer.x + (toLayer.x - fromLayer.x) * pulseFraction;
                // Spread a few pulses along the Y axis for each node
                nodes[pulseTargetIdx].forEach((n, ni) => {
                    const progress = Math.max(0, Math.min(1, pulseFraction + (ni * 0.05) - 0.2));
                    const py = nodes[pulseTargetIdx-1][ni % nodes[pulseTargetIdx-1].length].y +
                               (n.y - nodes[pulseTargetIdx-1][ni % nodes[pulseTargetIdx-1].length].y) * progress;
                    const brightness = n.act;
                    ctx.beginPath();
                    ctx.arc(px, py, 4, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(139, 92, 246, ${0.3 + brightness * 0.7})`;
                    ctx.shadowColor = 'rgba(139,92,246,1)';
                    ctx.shadowBlur = 12;
                    ctx.fill();
                    ctx.shadowBlur = 0;
                });

                if (pulseFraction >= 1) {
                    pulseFraction = 0;
                    pulseTargetIdx++;
                    if (pulseTargetIdx >= LAYERS.length) {
                        // One full pass done
                        loopCount++;
                        pulseTargetIdx = 1;
                        
                        if (!hasCascadedFC && loopCount >= 1) {
                            hasCascadedFC = true;
                            document.getElementById('fc-status').textContent = 'Forward pass complete! Sending scores to Softmax...';
                            setTimeout(runSoftmaxLayer, 1200);
                        }
                    }
                }

                requestAnimationFrame(fcLoop);
            }

            setTimeout(fcLoop, 800);

            // Hover tooltip
            const wrap = document.querySelector('.fc-network-wrap');
            wrap.addEventListener('mousemove', e => {
                const rect = canvas.getBoundingClientRect();
                const scaleX = W / rect.width;
                const mx = (e.clientX - rect.left) * scaleX;
                const my = (e.clientY - rect.top) * (H / rect.height);
                let hit = null;
                nodes.forEach(layer => {
                    layer.forEach(n => {
                        if (Math.hypot(mx - n.x, my - n.y) < 12) hit = n;
                    });
                });
                if (hit) {
                    tooltip.style.display = 'block';
                    tooltip.style.left = (e.clientX - rect.left + 12) + 'px';
                    tooltip.style.top = (e.clientY - rect.top - 10) + 'px';
                    tooltip.textContent = `${hit.label} — activation: ${hit.act.toFixed(3)}`;
                } else {
                    tooltip.style.display = 'none';
                }
            });
            wrap.addEventListener('mouseleave', () => tooltip.style.display = 'none');

        }, 1200);
    }

    // ===== SOFTMAX OUTPUT (STEP 9) — IMAGENET CLASSIFIER =====
    function runSoftmaxLayer() {
        const conn9 = document.getElementById('step-connector-8-9');
        const step9 = document.getElementById('step-9');
        conn9.style.display = 'flex';
        
        setTimeout(() => {
            step9.style.display = 'block';
            setTimeout(() => step9.classList.add('active-step'), 50);

            const barsContainer = document.getElementById('softmax-bars');
            barsContainer.innerHTML = '';

            const predictions = window.latestPredictions || [];
            
            if (predictions.length === 0) {
                barsContainer.innerHTML = '<p>No predictions found. Please process an image first.</p>';
                return;
            }

            const winner = predictions[0];

            predictions.forEach((p, i) => {
                const isWinner = i === 0;
                const pct = (p.probability * 100).toFixed(1);
                
                // Capitalize first letter of class name
                const className = p.className.charAt(0).toUpperCase() + p.className.slice(1);
                
                const row = document.createElement('div');
                row.className = 'softmax-bar-row';
                row.title = `${className}: ${pct}%`;
                row.innerHTML = `
                    <div class="softmax-bar-label" style="text-overflow: ellipsis; overflow: hidden; white-space: nowrap; width: 120px;">${className}</div>
                    <div class="softmax-bar-track">
                        <div class="softmax-bar-fill ${isWinner ? 'winner' : ''}" id="bar-fill-${i}" style="width:0%"></div>
                    </div>
                    <div class="softmax-bar-pct" id="bar-pct-${i}">0%</div>`;
                barsContainer.appendChild(row);
                
                row.addEventListener('mouseenter', () => {
                    row.querySelector('.softmax-bar-pct').textContent = `${pct}%`;
                });
            });

            // Animate bars staggered
            predictions.forEach((p, i) => {
                const pct = (p.probability * 100).toFixed(1);
                setTimeout(() => {
                    const fill = document.getElementById(`bar-fill-${i}`);
                    const label = document.getElementById(`bar-pct-${i}`);
                    if (fill) fill.style.width = `${pct}%`;
                    if (label) label.textContent = `${pct}%`;
                }, 200 + i * 150);
            });

            // Prediction banner
            setTimeout(() => {
                const conf = (winner.probability * 100).toFixed(1);
                const className = winner.className.charAt(0).toUpperCase() + winner.className.slice(1);
                const classEl = document.getElementById('softmax-class');
                if (classEl) {
                    classEl.textContent = className;
                    // Adjust font size if it's too long
                    if (className.length > 15) classEl.style.fontSize = '1.5rem';
                }
                
                const confEl = document.getElementById('softmax-conf');
                if (confEl) {
                    confEl.textContent = `Confidence: ${conf}% ✅ Real-Time CNN`;
                }
            }, 200 + predictions.length * 150 + 400);

        }, 1200);
    }

});
