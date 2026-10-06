document.addEventListener("DOMContentLoaded", () => {
    let currentMode = "sample"; // 'sample' or 'upload'
    let selectedFile = null;
    let showcaseSamples = [];

    // DOM Elements
    const btnModeSample = document.getElementById("btn-mode-sample");
    const btnModeUpload = document.getElementById("btn-mode-upload");
    const sampleInputView = document.getElementById("sample-input-view");
    const uploadInputView = document.getElementById("upload-input-view");
    const sampleSelect = document.getElementById("sample-select");
    const sampleMetaCard = document.getElementById("sample-metadata-card");
    const sampleRoleTag = document.getElementById("sample-role-tag");
    const sampleGt = document.getElementById("sample-gt");
    const sampleMethod = document.getElementById("sample-method");

    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const selectedFileName = document.getElementById("selected-file-name");
    const btnAnalyze = document.getElementById("btn-analyze");

    const progressArea = document.getElementById("progress-area");
    const progressFill = document.getElementById("progress-fill");
    const progressStatus = document.getElementById("progress-status");

    const resultSection = document.getElementById("result-section");
    const predCard = document.getElementById("pred-card");
    const predValue = document.getElementById("pred-value");
    const probReal = document.getElementById("prob-real");
    const probFake = document.getElementById("prob-fake");
    const decisionEval = document.getElementById("decision-eval");
    const nearThresholdCallout = document.getElementById("near-threshold-callout");
    const nearThresholdText = document.getElementById("near-threshold-text");

    const resultVideoPlayer = document.getElementById("result-video-player");
    const resFrames = document.getElementById("res-frames");
    const resFaces = document.getElementById("res-faces");
    const resFallback = document.getElementById("res-fallback");
    const resTime = document.getElementById("res-time");

    const expanderToggle = document.getElementById("expander-toggle");
    const expanderContent = document.getElementById("expander-content");
    const expanderArrow = document.getElementById("expander-arrow");
    const cropsGrid = document.getElementById("crops-grid");
    const freqGrid = document.getElementById("freq-grid");
    const showcaseGrid = document.getElementById("showcase-grid");

    // Mode Toggle
    btnModeSample.addEventListener("click", () => {
        currentMode = "sample";
        btnModeSample.classList.add("active");
        btnModeUpload.classList.remove("active");
        sampleInputView.style.display = "block";
        uploadInputView.style.display = "none";
    });

    btnModeUpload.addEventListener("click", () => {
        currentMode = "upload";
        btnModeUpload.classList.add("active");
        btnModeSample.classList.remove("active");
        sampleInputView.style.display = "none";
        uploadInputView.style.display = "block";
    });

    // Dropzone Events
    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    function handleSelectedFile(file) {
        selectedFile = file;
        selectedFileName.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
        selectedFileName.style.display = "block";
    }

    // Expander Toggle
    expanderToggle.addEventListener("click", () => {
        const isHidden = expanderContent.style.display === "none" || expanderContent.style.display === "";
        expanderContent.style.display = isHidden ? "block" : "none";
        expanderArrow.textContent = isHidden ? "▲" : "▼";
    });

    // Fetch Showcase Samples
    async function loadShowcaseSamples() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            showcaseSamples = data.samples || [];

            sampleSelect.innerHTML = "";
            showcaseGrid.innerHTML = "";

            showcaseSamples.forEach((s) => {
                // Populate dropdown
                const opt = document.createElement("option");
                opt.value = s.clip_filename;
                opt.textContent = `${s.clip_filename} [${s.ground_truth}]`;
                sampleSelect.appendChild(opt);

                // Populate showcase cards
                const card = document.createElement("div");
                card.className = "showcase-card";
                const tagClass = s.role.includes("False Positive")
                    ? "tag-fp"
                    : s.role.includes("Near-threshold")
                    ? "tag-near"
                    : "tag-correct";

                card.innerHTML = `
                    <span class="${tagClass}">${s.role.toUpperCase()}</span>
                    <div style="font-size: 12.5px; font-weight: 600; color: #F3F6FA; font-family: var(--font-mono); margin: 5px 0;">
                        ${s.clip_filename}
                    </div>
                    <div style="font-size: 12px; color: #9AA7B5; margin-bottom: 3px;">
                        Ground truth: <b style="color: #F3F6FA;">${s.ground_truth}</b> &nbsp;•&nbsp; 
                        Prediction: <b style="color: #F3F6FA;">${s.predicted_label}</b> &nbsp;•&nbsp; 
                        P(fake): <b style="color: #F3F6FA;">${(s.p_fake * 100).toFixed(1)}%</b>
                    </div>
                    <div style="font-size: 11px; color: #687585; line-height: 1.35;">
                        ${s.research_notes}
                    </div>
                `;

                // Clicking a showcase card selects and analyzes it
                card.addEventListener("click", () => {
                    sampleSelect.value = s.clip_filename;
                    btnModeSample.click();
                    updateSampleMetadata();
                    runAnalysis();
                });

                showcaseGrid.appendChild(card);
            });

            if (showcaseSamples.length > 0) {
                sampleSelect.value = showcaseSamples[0].clip_filename;
                updateSampleMetadata();
            }
        } catch (e) {
            console.error("Failed to load showcase samples:", e);
        }
    }

    function updateSampleMetadata() {
        const selected = showcaseSamples.find((s) => s.clip_filename === sampleSelect.value);
        if (selected) {
            sampleMetaCard.style.display = "block";
            sampleRoleTag.textContent = selected.role;
            sampleRoleTag.className = selected.role.includes("False Positive")
                ? "tag-fp"
                : selected.role.includes("Near-threshold")
                ? "tag-near"
                : "tag-correct";
            sampleGt.textContent = selected.ground_truth;
            sampleMethod.textContent = selected.manipulation_method;
        } else {
            sampleMetaCard.style.display = "none";
        }
    }

    sampleSelect.addEventListener("change", updateSampleMetadata);

    // Run Analysis Flow
    async function runAnalysis() {
        btnAnalyze.disabled = true;
        progressArea.style.display = "block";
        progressFill.style.width = "10%";
        progressStatus.textContent = "Stage 1/8: Initializing video stream...";

        try {
            let resData;
            if (currentMode === "sample") {
                const sampleName = sampleSelect.value;
                if (!sampleName) {
                    alert("Please select a valid research sample.");
                    resetProgress();
                    return;
                }

                // Simulate progressive stage feedback
                progressFill.style.width = "35%";
                progressStatus.textContent = "Stage 3/8: Detecting faces & extracting FFT frequency maps...";

                const res = await fetch("/api/analyze-sample", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ filename: sampleName }),
                });

                progressFill.style.width = "75%";
                progressStatus.textContent = "Stage 6/8: Computing temporal differences & Transformer pass...";

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || "Analysis failed");
                }
                resData = await res.json();
            } else {
                if (!selectedFile) {
                    alert("Please upload a video file first.");
                    resetProgress();
                    return;
                }

                progressFill.style.width = "30%";
                progressStatus.textContent = "Stage 2/8: Uploading & decoding 16 uniform frames...";

                const formData = new FormData();
                formData.append("file", selectedFile);

                const res = await fetch("/api/analyze-upload", {
                    method: "POST",
                    body: formData,
                });

                progressFill.style.width = "80%";
                progressStatus.textContent = "Stage 7/8: Executing dual-domain classification...";

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || "Analysis failed");
                }
                resData = await res.json();
                resData.video_url = URL.createObjectURL(selectedFile);
            }

            progressFill.style.width = "100%";
            progressStatus.textContent = "Stage 8/8: Inference complete!";
            setTimeout(() => {
                progressArea.style.display = "none";
                renderResults(resData);
            }, 300);
        } catch (err) {
            alert(`Inference Error: ${err.message}`);
            resetProgress();
        } finally {
            btnAnalyze.disabled = false;
        }
    }

    function resetProgress() {
        btnAnalyze.disabled = false;
        progressArea.style.display = "none";
    }

    // Render Inference Results
    function renderResults(res) {
        resultSection.style.display = "block";

        const isReal = res.prediction === "REAL";
        predCard.className = isReal ? "pred-card-real" : "pred-card-fake";
        predValue.className = isReal ? "pred-value-real" : "pred-value-fake";
        predValue.textContent = res.prediction;

        const fakePct = (res.fake_probability * 100).toFixed(1);
        const realPct = (res.real_probability * 100).toFixed(1);
        const threshPct = (res.decision_threshold * 100).toFixed(1);
        probReal.textContent = `${realPct}%`;
        probFake.textContent = `${fakePct}%`;

        const relSymbol = res.fake_probability > res.decision_threshold ? ">" : "<";
        decisionEval.textContent = `P(Fake) = ${fakePct}% ${relSymbol} ${threshPct}% → ${res.prediction}`;

        // Near-threshold State
        if (res.is_near_threshold) {
            const margin = Math.abs(res.margin_pp).toFixed(1);
            const dir = res.fake_probability <= res.decision_threshold ? "below" : "above";
            nearThresholdText.textContent = `Fake probability is ${margin} percentage points ${dir} the ${threshPct}% decision threshold.`;
            nearThresholdCallout.style.display = "block";
        } else {
            nearThresholdCallout.style.display = "none";
        }

        // Video & Summary
        if (res.video_url) {
            resultVideoPlayer.src = res.video_url;
            resultVideoPlayer.load();
        }
        resFrames.textContent = `${res.num_sampled_frames} uniform frames`;
        resFaces.textContent = `${res.faces_detected} / ${res.num_sampled_frames}`;
        resFallback.textContent = `${res.fallback_crops}`;
        resTime.textContent = `${res.inference_time_seconds.toFixed(2)} s`;

        // Tensor Visualizations
        cropsGrid.innerHTML = "";
        (res.face_crops_b64 || []).forEach((b64, idx) => {
            const div = document.createElement("div");
            div.className = "tensor-grid-item";
            div.innerHTML = `<img src="${b64}" alt="T=${idx + 1}"><div class="tensor-caption">T=${idx + 1}</div>`;
            cropsGrid.appendChild(div);
        });

        freqGrid.innerHTML = "";
        (res.freq_maps_b64 || []).forEach((b64, idx) => {
            const div = document.createElement("div");
            div.className = "tensor-grid-item";
            div.innerHTML = `<img src="${b64}" alt="FFT ${idx + 1}"><div class="tensor-caption">FFT ${idx + 1}</div>`;
            freqGrid.appendChild(div);
        });

        // Smooth scroll to results
        resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    btnAnalyze.addEventListener("click", runAnalysis);

    // Initialize
    loadShowcaseSamples();
});
