
# Bridging the Novice-Expert Divide: A Theory-Driven Approach to Visualising Sentiment in E-commerce Reviews

This repository contains the source code for the interactive dashboard artifact developed for the MSc dissertation, "Bridging the Novice-Expert Divide."

> **Abstract:** Online shopping reviews present a paradox of data abundance but insight scarcity, a challenge compounded by the novice-expert divide in the usability of analytical tools. This dissertation confronts this problem through a Design Science Research approach, documenting the design, implementation, and evaluation of an interactive dashboard. The artifact's design is grounded in a novel synthesis of theoretical frameworks: it operationalizes Shneiderman's Visual Information-Seeking Mantra to manage cognitive load through progressive disclosure, employs Vygotsky's theory of instructional scaffolding to enhance learnability, and is built upon a foundation of Universal Design to ensure equitable access. The research contributes not only an empirically validated artifact but also a critical analysis of the accessibility and usability trade-offs inherent in popular rapid-prototyping toolchains, offering a framework for more inclusive and effective data tool design. 

-----

## ✨ Key Features

The dashboard provides a multi-faceted approach to analyzing e-commerce review data:

  * **Interactive Product Search:** A universal gateway to search and filter products by category, title, average rating, and review count.
  * **High-Level Sentiment Overview:** An "at-a-glance" summary page featuring a 2x2 grid of Key Performance Indicators (KPIs), including novel metrics like "Reviewer Consensus" and "Verified Purchase Rate". 
  * **Progressive Disclosure Architecture:** The user journey is structured according to Shneiderman's mantra ("Overview first, zoom and filter, then details-on-demand"), guiding users from high-level summaries to granular insights in a cognitively manageable sequence. 
  * **Deep-Dive Analysis Pages:** Specialized pages for:
      * **Review Explorer:** Read, sort, and search through individual reviews. 
      * **Keyword & Phrase Analysis:** Analyze the sentiment and trends associated with specific terms. 
      * **Aspect Analysis:** Investigate sentiment towards specific product features (e.g., "battery life," "screen quality"). 
      * **Product Comparison:** A side-by-side comparative view of two products across multiple metrics and features. 
  * **Novel Visualizations:** Includes an interactive "Mismatched Reviews" scatter plot to identify anomalous or sarcastic reviews. 
  * **Instructional Scaffolding:** "Designed-in" help is provided via "ⓘ" popover icons next to every chart and novel metric, explaining what the visualization shows, how to use it, and what insights can be learned. 

-----

## 🛠️ Technology Stack

  * **Web Framework:**([https://streamlit.io/](https://streamlit.io/)) 
  * **Database:**(https://duckdb.org/) 
  * **Data Manipulation:** [Pandas](https://pandas.pydata.org/) 
  * **NLP & Sentiment Analysis:**
      * ([https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)) (using the `cardiffnlp/twitter-roberta-base-sentiment` model) 
      * (https://github.com/yangheng95/PyABSA) for Aspect-Based Sentiment Analysis 
  * **Data Visualization:** [Altair](https://altair-viz.github.io/), [Plotly](https://plotly.com/python/), [Matplotlib](https://matplotlib.org/),([https://github.com/amueller/word\_cloud](https://github.com/amueller/word_cloud)) 

-----

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

  * Python 3.9+
  * Git

### Installation & Setup

1.  \*\*Clone the repository:\*\*bash
    git clone [https://github.com/wxxthiq/SA-Amazon-Reviews.git](https://www.google.com/search?q=https://github.com/wxxthiq/SA-Amazon-Reviews.git)
    cd SA-Amazon-Reviews

    ```
    
    ```

2.  **Create and activate a virtual environment (recommended):**

      * **On macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
      * **On Windows:**
        ```bash
        python -m venv venv
        ```

    .\\venv\\Scripts\\activate
    \`\`\`

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    This will install all the necessary Python packages listed in the `requirements.txt` file. 

### Data Setup

The application is designed to automatically download the required database file from Kaggle on its first run. This requires you to have a Kaggle account and an API token.

1.  **Get your Kaggle API Token:**

      * Log in to your Kaggle account.
      * Go to your account settings page (`https://www.kaggle.com/your-username/account`).
      * Click on the "Create New API Token" button. This will download a `kaggle.json` file containing your username and API key.

2.  **Create a Streamlit Secrets file:**

      * In the root of the project directory, create a new folder named `.streamlit`.
      * Inside the `.streamlit` folder, create a new file named `secrets.toml`.
      * Add your Kaggle credentials to this file in the following format:
        ```toml
        [kaggle]
        username = "YOUR_KAGGLE_USERNAME"
        key = "YOUR_KAGGLE_API_KEY"
        ```
      * Replace `"YOUR_KAGGLE_USERNAME"` and `"YOUR_KAGGLE_API_KEY"` with the values from your `kaggle.json` file.

The application will use these credentials to authenticate with Kaggle and download the `amazon_reviews_final.duckdb` database. 

-----

## ▶️ Running the Application

Once the setup is complete, you can run the Streamlit application with the following command from the project's root directory:

```bash
streamlit run app.py
```

The application should open automatically in your web browser.

-----

## 📂 Repository Structure

```
.
├──.streamlit/
│   └── secrets.toml    # Kaggle API credentials (you must create this)
├── pages/
│   ├── 1_Sentiment_Overview.py
│   ├── 2_Review_Explorer.py
│   ├── 3_Keyword_Analysis.py
│   ├── 4_Aspect_Analysis.py
│   └── 5_Product_Comparison.py
├── utils/
│   └── database_utils.py # Functions for database connection and queries
├── app.py                # Main entry point and product search page
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

-----

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

-----

## 🎓 Citation

If you use this work for academic purposes, please cite the dissertation:

> Soualhi, M. W. (2025). *Bridging the Novice-Expert Divide: A Theory-Driven Approach to Visualising Sentiment in E-commerce Reviews* (MSc Dissertation). The University of Manchester, Manchester, UK.

```
