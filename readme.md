##### How to run the project

###### Prerequisites
- Python 3.x installed on your system.

##### Setup Instructions

1. Clone the repository:
    ```bash
    git clone git@github.com:Uchswas/CSC791-Assignment3.git
    cd CSC791-Assignment3
    ```

2. **Create a Virtual Environment**  
   Run the following command to create a virtual environment named `venv`:
    ```bash
    python -m venv venv_genai
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
    For generating embeddings, run the command. It will generate embeddings that can be found in the `results/code_embeddings` folde
    ```bash
    python3 embedding.py
    ```
    To generate results from embeddings, run the command. This will produce images saved in the `results/images` folder and distances in the `results/text_files` folder.
    ```bash
    python3 embedding.py
    ```


