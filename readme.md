# How to run the project

## Prerequisites
- Python 3.x installed on your system.

## Setup Instructions

1. **Clone the repository** (if not already cloned):
    ```bash
    git clone git@github.com:Uchswas/CSC791-Assignment3.git
    cd CSC791-Assignment3
    ```

2. **Create a Virtual Environment**  
   Run the following command to create a virtual environment named `venv`:
    ```bash
    python -m venv venv
    ```

3. **Activate the Virtual Environment**

   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install Required Packages**  
   Use `pip` to install all dependencies listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the Project**  
   Finally, execute the main script:
    ```bash
    python main.py
    ```

## Deactivating the Virtual Environment
When you're done, deactivate the virtual environment:
```bash
deactivate
