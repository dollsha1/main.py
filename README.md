
### 1. Extract and index metadata

This command performs the following steps:

- Connects to **Azure Database for MySQL**
- Extracts table and column metadata from the database
- Computes basic column profiling statistics
- Generates vector embeddings using **Azure OpenAI**
- Creates or updates the **Azure AI Search** index
- Uploads vectorized documents to the search index

Run the command:

python main.py extract_index

### 2. Search indexed metadata

This command performs semantic search on the indexed database metadata.

The following steps are executed:
- Converts the user query into a vector embedding using **Azure OpenAI**
- Executes a vector similarity search against **Azure AI Search**
- Returns the most relevant tables and columns

Run the command:

python main.py search "<your_query>"

Example:

python main.py search "customer email address"

