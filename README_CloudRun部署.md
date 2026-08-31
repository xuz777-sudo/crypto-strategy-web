# Cloud Run 部署補丁

請把這 3 個項目上傳到 GitHub repository `crypto-strategy-web` 根目錄：

- `Dockerfile`
- `.dockerignore`
- `pages/6_資料診斷.py`

然後回 Google Cloud Run：

1. 點「連結存放區」。
2. 選 GitHub。
3. 選 `xuz777-sudo/crypto-strategy-web`。
4. Branch 選 `main`。
5. Build 類型使用 Dockerfile。
6. Region 選 `asia-east1`。
7. Container port 使用 `8080`。
8. Authentication 選允許未驗證存取，網站才可直接從瀏覽器開啟。
9. 部署完成後先進入「資料診斷」，執行完整診斷。
