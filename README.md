# Gallstone Prediction Project

![License](https://img.shields.io/github/license/Bhawesh-Agrawal/Gallstone_Risk_Stratification?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=next.js)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-F37626?style=for-the-badge&logo=jupyter)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Visitor Badge](https://visitor-badge.laobi.icu/badge?page_id=your-username.gallstone-prediction)
![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red?style=for-the-badge)

---

## About

A full-stack project to predict the presence of gallstones, based on a medical dataset from a university experiment.  
Key components:
- **Notebook:** Contains all exploratory data analysis (EDA) and modeling experiments.
- **Backend:** FastAPI server with Gemini integration, serving prediction and a fine-tuned chatbot for gallstone-related questions.
- **Frontend:** Next.js app (in `frontend/gallstone`), using Cloudinary, Convex, and a Hugging Face API to interact with backend and website.

---

## Features

- Predict gallstone risk from input data
- Chatbot for answering medical queries about gallstones
- Modern web stack: FastAPI, Next.js, Jupyter
- Integrates Cloudinary (media), Convex (realtime data), Hugging Face API (ML/AI)[web:26][web:23]

---

## Project Structure

```
.
├── backend/                # FastAPI app and API
├── frontend/
│   └── gallstone/          # Next.js frontend
├── notebook/               # Experiments & EDA notebooks
├── README.md               # This document
```

---

## Getting Started

### 1. Clone the repository

```
git clone https://github.com/Bhawesh-Agrawal/Gallstone_Risk_Stratification.git
cd gallstone-prediction
```

### 2. Setup Environment Variables

Create `.env` files with your credentials for both backend and frontend.

- **backend/.env**
  ```
  FASTAPI_HOST=0.0.0.0
  FASTAPI_PORT=8000
  HUGGINGFACE_API_KEY=your_huggingface_api_key
  GOOGLE_GEMINI_API_KEY=your_gemini_api_key
  ```
- **frontend/gallstone/.env.local**
  ```
  NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
  NEXT_PUBLIC_CLOUDINARY_API_KEY=your_cloudinary_api_key
  NEXT_PUBLIC_CONVEX_URL=your_convex_url
  ```

### 3. Run Backend

```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run Frontend

```
cd frontend/gallstone
npm install
npm run dev
```

### 5. Open Notebook

```
cd notebook
jupyter notebook
```

---

## Usage

- Open `http://localhost:3000` in browser for the website interface.
- Try model prediction, ask chatbot questions.
- Use Jupyter notebooks for exploring data, building and testing models.

---

## Technologies

- FastAPI, Gemini API, Hugging Face, Python (Backend)
- Next.js, Cloudinary, Convex (Frontend)
- Jupyter Notebook, pandas, scikit-learn (Experiments)

---

## Disclaimer

For educational and demonstration purposes.  
No prediction or chatbot answer should be considered as medical advice. Always consult certified medical professionals.

---

## License

Distributed under the MIT License.

```

This file follows modern open-source standards and clean formatting.[1][3][5][7]

[1](https://github.com/othneildrew/Best-README-Template)
[2](https://github.com/fastapi/full-stack-fastapi-template)
[3](https://coding-boot-camp.github.io/full-stack/github/professional-readme-guide/)
[4](https://dev.to/thepiyushmalhotra/how-to-design-an-attractive-github-profile-readme-1ppg)
[5](https://github.com/matiassingers/awesome-readme)
[6](https://www.reddit.com/r/learnprogramming/comments/vxfku6/how_to_write_a_readme/)
[7](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)
[8](https://readmi.xyz/templates)
[9](https://github.com/topics/full-stack-project)
