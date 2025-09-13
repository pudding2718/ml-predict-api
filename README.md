# Bitus Labs Machine Learning Engineer

A behavior prediction API implemented with **Java Spring Boot + ONNX Runtime**, exposing the endpoint `POST /predict_behavior`.
**Nginx** is used to terminate TLS (HTTPS). The backend container serves HTTP on port 8080, and the model is provided via a bind mount. The training workflow uses **Python (PyTorch)** and is placed in `model-dev/script.ipynb`.

```
Client ──HTTPS(443)──> Nginx ──HTTP──> Spring Boot (8080) ──> ONNXRuntime ──> model.onnx
```

- **Docker Hub**: `pudding2718/ml-predict-api:latest`
- **GitHub Link:** https://github.com/pudding2718/ml-predict-api.git

---
## Table of Contents

- [Quick Start](https://chatgpt.com/c/68c4e147-8edc-8331-a5c2-02d81323086c#快速开始)
- [Environment Setup & Start Commands](https://chatgpt.com/c/68c4e147-8edc-8331-a5c2-02d81323086c#环境建置与启动指令)
- [API Docs & Test Samples](https://chatgpt.com/c/68c4e147-8edc-8331-a5c2-02d81323086c#api-文件与测试样例)
- [Model Training (Python / Jupyter)](https://chatgpt.com/c/68c4e147-8edc-8331-a5c2-02d81323086c#模型训练python--jupyter)
- [Git Branches](https://chatgpt.com/c/68c4e147-8edc-8331-a5c2-02d81323086c#git-分支说明)
- [Deployment Guide (with HTTPS)](https://chatgpt.com/c/68c4e147-8edc-8331-a5c2-02d81323086c#部署手册含-https)
- [FAQ](https://chatgpt.com/c/68c4e147-8edc-8331-a5c2-02d81323086c#常见问题)
- [Author & Reflections](https://chatgpt.com/c/68c4e147-8edc-8331-a5c2-02d81323086c#作者简介与开发心得)

---

## Quick Start

> Prerequisite: `model/model.onnx` exists on the host (not committed to Git; it is mounted at deployment time).

**1) Pull the image**

```bash
docker pull pudding2718/ml-predict-api:latest
```

**2) Run API only (without HTTPS, convenient for debugging)**

```bash
# Linux/Mac
docker run -d --name ml-api \
  -p 18080:8080 \
  -v "$(pwd)/model:/root/Bitus-Labs:ro" \
  <dockerhub-username>/ml-predict-api:latest
# Windows PowerShell
docker run -d --name ml-api `
  -p 18080:8080 `
  -v "$((Get-Location).Path)/model:/root/Bitus-Labs:ro" `
  <dockerhub-username>/ml-predict-api:latest
```

**3) Test request**

```bash
curl -i -H "Content-Type: application/json" \
  --data-binary '{"input":[[1,2,15,3,1,2,15,3]]}' \
  http://localhost:18080/predict_behavior
```

---

## Environment Setup & Start Commands

**Dependencies**

- Docker 24+ / Docker Desktop
- (Optional for local development) JDK 17, Gradle Wrapper

**Image Notes**

- Base image: `mcr.microsoft.com/openjdk/jdk:17-ubuntu` (glibc, compatible with ONNX Runtime)
- Runtime deps: `libstdc++6`, `libgomp1`
- Spring Boot: `8080` (HTTP)
- Model path inside the container: `/root/Bitus-Labs/model.onnx` (provided by bind mount)

**Local run (optional)**

```bash
cd api-java
./gradlew bootRun   # Use gradlew.bat on Windows
```

---

## API Docs & Test Samples

**Endpoint**

- `POST /predict_behavior`
- `Content-Type: application/json`

**Request body (2D array: one sample per row, fixed feature order)**

```json
{
  "input": [
    [visitorid, itemid, hour, dayofweek]
  ]
}
```

**Example**

```json
{ "input": [[1, 2, 15, 3, 1, 2, 15, 3]] }
```

**Common error examples**

- Missing `input`:

  ```json
  {
    "error_type": "IllegalArgumentException",
    "error": "Input data must contain 'input' field",
    "status": "failed"
  }
  ```

- `input` is not a 2D array:

  ```json
  {
    "error_type": "IllegalArgumentException",
    "error": "Each input row must be a list",
    "status": "failed"
  }
  ```

---

## Model Training (Python / Jupyter)

The training code is in `model-dev/script.ipynb`. Core steps:

**1) Environment & dependencies**

```bash
# Recommended Python 3.10+
pip install pandas scikit-learn torch matplotlib onnx
# Optional: onnxruntime (for local testing)
pip install onnxruntime
```

**2) Data preparation**

- Raw file: `model-dev/events.csv` (example columns: `visitorid,itemid,event,timestamp,transactionid,...`)

- Preprocessing highlights (excerpt from the notebook):

  ```python
  import pandas as pd
  from sklearn.preprocessing import LabelEncoder

  data = pd.read_csv('events.csv')
  data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

  le_user, le_item = LabelEncoder(), LabelEncoder()
  data['visitorid'] = le_user.fit_transform(data['visitorid'])
  data['itemid']    = le_item.fit_transform(data['itemid'])

  data['event'] = data['event'].map({'view':0, 'cart':1, 'purchase':2}).fillna(0).astype(int)
  data['transactionid'] = data['transactionid'].fillna(0)

  data['hour'] = data['timestamp'].dt.hour
  data['dayofweek'] = data['timestamp'].dt.dayofweek

  features = ['visitorid','itemid','hour','dayofweek']  # Feature order must match inference
  X = data[features].values
  y = data['event'].values
  ```

> ⚠️ If you used **StandardScaler** during training, keep the **same transform** at inference (save & load the same scaler); if not implemented yet, consider skipping normalization to avoid distribution mismatch between training and inference.

**3) Model & training (LSTM + Dense)**

```python
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split

class UserBehaviorModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)          # x: [B, T=1, F=input_size]
        out = self.fc(out[:, -1, :])   # take the last time step
        return out

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # [B,1,F]
y_train_t = torch.tensor(y_train, dtype=torch.long)

model = UserBehaviorModel(input_size=len(features))
crit = nn.CrossEntropyLoss()
opt  = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(200):
    model.train()
    opt.zero_grad()
    loss = crit(model(X_train_t), y_train_t)
    loss.backward()
    opt.step()
```

**4) Evaluation & save**

```python
# Evaluation omitted (accuracy / f1)
torch.save(model.state_dict(), 'user_behavior_model.pth')
```

**5) Export ONNX (key: dummy_input shape must match the input)**

> You have 4 input features, so `dummy_input` should be **(1,1,4)** (not 8).

```python
import torch, torch.onnx

model = UserBehaviorModel(input_size=len(features))
model.load_state_dict(torch.load('user_behavior_model.pth', map_location='cpu'))
model.eval()

dummy_input = torch.randn(1, 1, len(features))  # (batch=1, seq_len=1, input_size=4)

torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0:"batch"}, "logits": {0:"batch"}},
    opset_version=17, verbose=False
)
```

**6) Place in runtime directory**

- Put the exported `model.onnx` into the repo’s `model/` folder (`.gitignore` already excludes this directory).
- At runtime, mount it via: `-v "<host>/model:/root/Bitus-Labs:ro"`. The fixed path inside the container is `/root/Bitus-Labs/model.onnx`.

---

## Git Branches

- **`main`**: Documentation (README, guides, deployment manual). May also contain the full directory for reproducibility.
- **`api-java`**: Backend (Spring Boot + ONNX Runtime).
- **`docker`**: Containers & gateway (`docker/Dockerfile`, `docker/nginx.conf`, `docker/docker-compose.yml`).
- **`model-dev`**: Jupyter Notebook and training scripts (e.g., `script.ipynb`) plus lightweight results (avoid committing large binaries).

> `.gitignore` excludes: `model/`, `docker/certs/`, `api-java/libs/`, build artifacts, etc. ONNX Runtime is brought in via Gradle dependencies to avoid committing large JARs.

---

## Deployment Guide (with HTTPS)

**1) Generate self-signed certs (testing only)**

```powershell
@'
[req]
default_bits=2048
prompt=no
default_md=sha256
x509_extensions=v3_req
distinguished_name=dn
[dn]
C=CN
ST=Local
L=Local
O=Local
OU=Dev
CN=localhost
[v3_req]
subjectAltName=@alt_names
[alt_names]
DNS.1=localhost
IP.1=127.0.0.1
'@ | Set-Content -Encoding ascii docker/certs/localhost.cnf

openssl req -x509 -nodes -days 365 -newkey rsa:2048 `
  -keyout docker/certs/server.key -out docker/certs/server.crt `
  -config docker/certs/localhost.cnf
```

**2) Compose up**

```bash
cd docker
docker compose up -d
```

**3) HTTPS verification (self-signed requires -k)**

```powershell
@'
{"input":[[1,2,15,3,1,2,15,3]]}
'@ | Out-File body.json -Encoding ascii
curl.exe --tlsv1.2 --ssl-no-revoke -k https://localhost:443/predict_behavior `
  -H "Content-Type: application/json" --data-binary "@body.json"
```

---

## FAQ

- **502 Bad Gateway (Nginx)**: path rewrite issue. Ensure:

  ```nginx
  location /predict_behavior { proxy_pass http://app_upstream; }
  ```

- **Upstream `app:8080` not found**: make sure the compose service is named `app` and use `depends_on` to start it first.

- **`libstdc++.so.6` missing**: use an Ubuntu/Temurin base image and install `libstdc++6 libgomp1`.

- **Model not found**: verify the bind mount `-v "<host>/model:/root/Bitus-Labs:ro"` and that `/root/Bitus-Labs/model.onnx` exists inside the container.

- **Feature standardization**: if you standardized features during training, you must reproduce the same transform at inference; otherwise, remove standardization or implement it in the API.

---

## Author & Reflections

- Author: _pudding2718_
- Contact: [tinawang2718@gmail.com](mailto:tinawang2718@gmail.com)

**Reflections (expanded):**

- **ONNX Runtime & base image choice.** Alpine (musl) frequently runs into missing `libstdc++.so.6` and OpenMP issues with ONNX Runtime. Switching to an Ubuntu/Temurin (glibc) base and explicitly installing `libstdc++6` and `libgomp1` made the runtime stable and avoided low-level loader errors.
- **TLS termination at Nginx.** Offloading HTTPS to Nginx and keeping Spring Boot on plain HTTP (8080) simplified configuration and eliminated tricky double-TLS handshakes that had previously led to `502` errors and TLS negotiation failures.
- **Model & secrets kept out of Git.** The model (`model/`), certs (`docker/certs/`), and large binaries are ignored via `.gitignore`. This enforces clean Git history, reduces repo size, and simplifies security. Models are mounted into the container at runtime rather than baked into the image.
- **Dependency management via Gradle instead of fat JARs.** Large JARs (e.g., ONNX Runtime) are not committed. Using Gradle coordinates avoids GitHub’s 100 MB limit and keeps the repository portable. When a large file did slip in, history rewriting (filter-branch/filter-repo) was necessary to pass remote checks.
- **Windows specifics: paths, encoding, ports.** On Windows, volume mounts with spaces or non-ASCII characters require careful quoting. CRLF/LF differences are benign but noisy; a `.gitattributes` helps normalize. Also, port 443 can be occupied by security software; remapping to an alternate host port (e.g., `8443:443`) is a pragmatic workaround.
- **Health checks and service startup order.** If a health check is too strict or the app requires time to load the model, `depends_on` plus relaxed health thresholds prevents Nginx from starting before the app is actually ready. In development, you can temporarily remove health checks to focus on core functionality.

---
