#### RQs and key findings:
- RQ1: Can syntactic similarity matrices be solely relied upon for detecting code clones?
- RQ2: Can LLM models like CodeBERT outperform syntactical similarity metrics in identifying code clones?
- RQ3: Does the code embeddings of Codebert rely mostly on the syntax for clone detection

Full Documentation Link : https://docs.google.com/document/d/1A6qpW8EVIG2F48-WM9zwJ99KnaJkwlREFDU1AAGKUZs/edit?tab=t.0

#### Prerequisites
- Python 3.x installed on your system.

#### Setup Instructions

1. Clone the repository:
    ```bash
    git clone git@github.com:Uchswas/CSC791-Assignment3.git
    cd CSC791-Assignment3
    ```

2. **Create a Virtual Environment**  
   Run the following command to create a virtual environment named `venv`:
    ```bash
    python3 -m venv venv_genai
    ```

3. **Activate the Virtual Environment**

     ```bash
     source venv_genai/bin/activate
     ```

4. **Install Required Packages**  
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the Project**  
   First go to the `src` folder
    ```bash
    cd src
    ```
    For generating embeddings, run the command. It will generate embeddings that can be found in the `results/code_embeddings` folder
    ```bash
    python3 embedding.py
    ```
    To generate results from embeddings, run the command. This will produce images saved in the `results/images` folder and distances in the `results/text_files` folder.
    ```bash
    python3 vis.py
    ```


