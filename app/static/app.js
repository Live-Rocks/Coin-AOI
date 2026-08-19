const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const uploadButton = document.querySelector("#upload-button");
const exampleButton = document.querySelector("#example-button");
const selectedFile = document.querySelector("#selected-file");
const resultSection = document.querySelector("#result-section");
const resultTitle = document.querySelector("#result-title");
const resultStatus = document.querySelector("#result-status");
const resultMessage = document.querySelector("#result-message");
const resultContent = document.querySelector("#result-content");
const annotatedImage = document.querySelector("#annotated-image");
const modelName = document.querySelector("#model-name");
const confidenceThreshold = document.querySelector("#confidence-threshold");
const detectionCount = document.querySelector("#detection-count");
const detectionList = document.querySelector("#detection-list");

function setSelectedFile(file) {
  fileInput.files = file ? createFileList(file) : new DataTransfer().files;
  selectedFile.textContent = file ? `${file.name} · ${formatBytes(file.size)}` : "尚未選擇圖片";
  uploadButton.disabled = !file;
}

function createFileList(file) {
  const transfer = new DataTransfer();
  transfer.items.add(file);
  return transfer.files;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setLoading(message) {
  exampleButton.disabled = true;
  uploadButton.disabled = true;
  resultTitle.textContent = "正在執行推論";
  resultStatus.textContent = "推論中";
  resultStatus.className = "status is-loading";
  resultMessage.textContent = message;
  resultContent.hidden = true;
  resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function clearLoading() {
  exampleButton.disabled = false;
  uploadButton.disabled = !fileInput.files.length;
}

function setError(message) {
  resultTitle.textContent = "推論未完成";
  resultStatus.textContent = "無法完成";
  resultStatus.className = "status is-error";
  resultMessage.textContent = message;
  resultContent.hidden = true;
}

function setResult(payload) {
  resultTitle.textContent = "推論結果";
  resultStatus.textContent = "完成";
  resultStatus.className = "status is-success";
  resultMessage.textContent =
    payload.detection_count > 0
      ? "下方為模型在固定信心閾值下產生的偵測結果。"
      : "模型在固定信心閾值下沒有輸出偵測框；這不代表硬幣通過檢驗。";

  annotatedImage.src = payload.annotated_image_data_url;
  modelName.textContent = payload.model;
  confidenceThreshold.textContent = payload.confidence_threshold.toFixed(2);
  detectionCount.textContent = String(payload.detection_count);
  detectionList.replaceChildren();

  if (payload.detections.length === 0) {
    const item = document.createElement("li");
    item.className = "empty";
    item.textContent = "沒有偵測框";
    detectionList.append(item);
  } else {
    payload.detections.forEach((detection) => {
      const item = document.createElement("li");
      const label = document.createElement("span");
      const confidence = document.createElement("strong");
      label.textContent = detection.class_name;
      confidence.textContent = `${(detection.confidence * 100).toFixed(1)}%`;
      item.append(label, confidence);
      detectionList.append(item);
    });
  }

  resultContent.hidden = false;
}

async function requestPrediction(url, options = {}) {
  try {
    const response = await fetch(url, options);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "推論服務暫時無法使用。");
    }
    setResult(payload);
  } catch (error) {
    setError(error.message || "推論服務暫時無法使用。");
  } finally {
    clearLoading();
  }
}

fileInput.addEventListener("change", () => setSelectedFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) {
    setSelectedFile(file);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const [file] = fileInput.files;
  if (!file) {
    return;
  }
  setLoading(`正在分析 ${file.name}…`);
  const body = new FormData();
  body.append("image", file);
  await requestPrediction("/api/predict", { method: "POST", body });
});

exampleButton.addEventListener("click", async () => {
  setLoading("正在分析預設範例圖片…");
  await requestPrediction("/api/predict-example", { method: "POST" });
});
