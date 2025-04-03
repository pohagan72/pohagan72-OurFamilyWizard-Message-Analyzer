# OurFamilyWizard Message Analyzer

**Disclaimer:** This project is an independent initiative and is not affiliated with or endorsed by OurFamilyWizard.

## Overview

The OurFamilyWizard Message Analyzer is a Flask-based application designed to process and analyze message logs exported from the OurFamilyWizard platform. It provides a user-friendly web interface to:

1.  **Parse PDF Message Logs:** Accepts OurFamilyWizard message logs in PDF format.
2.  **Extract and Organize Messages:** Extracts text from the PDF, identifies participants, and organizes messages by year and month.
3.  **AI-Powered Analysis:** Utilizes either Google Gemini or Azure OpenAI to analyze message content, identify key discussion points, and detect mentions of specific topics like drugs, alcohol, and money, as well as overall sentiment and tone.
4.  **Generate Summary Reports:** Creates monthly and yearly summary reports in PDF format, highlighting key findings from the AI analysis.
5.  **Archive Reports:** Zips the generated reports into a single archive for easy download.
6.  **Cloud Storage Integration (Optional):** Can upload the generated reports to Azure Blob Storage for archival and access.

## Features

*   **Web-Based Interface:** Easy-to-use web interface for uploading files, selecting AI models, and generating reports.
*   **PDF Parsing:** Utilizes PyMuPDF to extract text content from OurFamilyWizard message log PDFs.
*   **Message Organization:** Parses and organizes messages by year and month for structured analysis.
*   **AI Analysis:** Leverages Google Gemini or Azure OpenAI to analyze message content and generate summaries.
*   **Report Generation:** Generates comprehensive monthly and yearly summary reports in PDF format using ReportLab.
*   **Azure Blob Storage Integration:** Option to upload generated reports to Azure Blob Storage for secure storage and retrieval.
*   **Error Handling and Logging:** Implements robust error handling and logging to ensure smooth operation and aid in troubleshooting.
*   **Configurable Settings:** Allows customization of AI model, API keys, and other parameters through environment variables.

## Technologies Used

*   **Flask:** A lightweight Python web framework for building the application.
*   **PyMuPDF:** A Python library for PDF processing.
*   **ReportLab:** A Python library for generating PDF documents.
*   **Google Gemini:** Google's AI model for natural language processing and analysis.
*   **Azure OpenAI:** Microsoft's OpenAI service for natural language processing and analysis.
*   **Azure Blob Storage:** Microsoft's cloud storage service for storing generated reports.
*   **HTML/CSS/JavaScript:** For creating the user interface.

## Requirements

*   Python 3.7+
*   The following Python packages (install using `pip install -r requirements.txt`):
    *   Flask
    *   PyMuPDF
    *   reportlab
    *   azure-storage-blob
    *   python-dotenv
    *   requests
    *   google-generativeai
    *   waitress

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone [repository_url]
    cd [repository_directory]
    ```

2.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Azure Blob Storage (Optional):**

    *   Set the `AZURE_STORAGE_CONNECTION_STRING` and `AZURE_STORAGE_CONTAINER_NAME` variables in the `app.py` file or as environment variables.

4.  **Configure Google Gemini API (Optional):**

    *   Set the `GOOGLE_API_KEY` variable in the `app.py` file or as an environment variable.

5.  **Configure Azure OpenAI API (Optional):**

    *   Set the `AZURE_OAI_ENDPOINT` and `AZURE_OAI_KEY` variables in the `app.py` file or as environment variables.

6.  **Run the application:**

    ```bash
    python app.py
    ```

    Alternatively, for production, you can use waitress:

    ```bash
    waitress-serve --listen=*:8000 app:app
    ```

7.  **Access the application:**

    Open your web browser and navigate to `http://localhost:5000` (or the appropriate address if running on a different port/server).

## Usage

1.  **Upload PDF:** Upload your OurFamilyWizard message log PDF file using the "Upload PDF" section on the main page.
2.  **Generate Reports:** After the PDF is processed, a section to generate reports will appear. Select the AI model you want to use (Google Gemini or Azure OpenAI) and click the "Generate Reports" button.
3.  **Download Reports:** Once the reports are generated, they will be available for download as a ZIP archive.  The page will automatically redirect, and a flash message will provide a download link.

## Configuration

The following configuration options are available:

*   `AZURE_STORAGE_CONNECTION_STRING`: Azure Blob Storage connection string.
*   `AZURE_STORAGE_CONTAINER_NAME`: Azure Blob Storage container name.
*   `GOOGLE_API_KEY`: Google Gemini API key.
*   `AZURE_OAI_ENDPOINT`: Azure OpenAI API endpoint.
*   `AZURE_OAI_KEY`: Azure OpenAI API key.
*   `AZURE_OAI_MAX_TOKENS`: Max tokens for Azure OpenAI response.
*   `MAX_RETRIES`: Maximum number of retries for failed AI API calls.
*   `INITIAL_WAIT_SECONDS`: Initial wait time before retrying API call.
*   `WAIT_INCREMENT_SECONDS`: Additional seconds to wait for each subsequent retry.
*   `MAX_INPUT_CHARS`: Maximum characters allowed as input to the AI model.
*   `MAX_WORKERS`: Number of parallel workers for concurrent tasks.

These variables can be set either directly in the `app.py` file or as environment variables.  Using environment variables is highly recommended for security, especially for API keys.

## Error Handling

The application includes comprehensive error handling and logging. If any errors occur during PDF processing, AI analysis, or report generation, they will be logged to the console and displayed as flash messages in the web interface.

## Contributing

Contributions to this project are welcome. Please submit a pull request with your proposed changes.

## License

[Specify License - e.g., MIT License]
