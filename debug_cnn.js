const fs = require('fs');
let ganPath = '/Users/angelrao/Documents/deepvision/legacy/gan.js';
let content = fs.readFileSync(ganPath, 'utf8');

// Inject debug canvas drawing right before prediction
let replacement = `
                    // Threshold background noise
                    t = t.toFloat().div(255);
                    t = t.sub(0.1).relu(); 
                    const maxVal = t.max();
                    t = t.div(maxVal.add(1e-5));
                    
                    // DEBUG: Draw tensor to screen so we can see what the CNN sees
                    let debugCanvas = document.getElementById('debug-cnn');
                    if (!debugCanvas) {
                        debugCanvas = document.createElement('canvas');
                        debugCanvas.id = 'debug-cnn';
                        debugCanvas.width = 56;
                        debugCanvas.height = 56;
                        debugCanvas.style = 'border:2px solid red; position:fixed; top:10px; right:10px; z-index:99999; background:black;';
                        document.body.appendChild(debugCanvas);
                    }
                    tf.browser.toPixels(t, debugCanvas).then(() => {
                        console.log("Debug canvas updated!");
                    });
                    
                    return t.expandDims(0);
                });
`;

content = content.replace(/t = t\.toFloat\(\)\.div\(255\);\s*t = t\.sub\(0\.1\)\.relu\(\);\s*const maxVal = t\.max\(\);\s*return t\.div\(maxVal\.add\(1e-5\)\)\.expandDims\(0\);\s*}\);/g, replacement);

fs.writeFileSync(ganPath, content);
console.log("Injected debug canvas!");
