# OurFamilyWizard Message Analyzer

![Flask](https://img.shields.io/badge/Flask-2.3.2-blue)
![Python](https://img.shields.io/badge/Python-3.8+-blue)

A web application for analyzing message logs from OurFamilyWizard (OFW) communication platform, generating structured reports with AI-powered insights.

**Important Note**: This is an independent open-source project and is not affiliated with or endorsed by OurFamilyWizard LLC. It was created to help users analyze their own message history from the platform.

## Features

- **PDF Processing**: Parses OFW message log PDFs to extract and organize messages by date
- **AI Analysis**: Uses Google Gemini or Azure OpenAI to analyze message content for:
  - Key discussion topics and summaries
  - Mentions of money, drugs, or alcohol
  - Sentiment and conflict analysis
- **Report Generation**: Creates comprehensive PDF reports:
  - Monthly message logs
  - Monthly analysis reports
  - Yearly summary reports
- **Cloud Integration**: Option to upload results to Azure Blob Storage

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pohagan72/OurFamilyWizard-Message-Analyzer.git
   cd OurFamilyWizard-Message-Analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Create a `.env` file with your API keys:
     ```
     GOOGLE_API_KEY=your_google_api_key
     AZURE_OAI_KEY=your_azure_openai_key
     AZURE_STORAGE_CONNECTION_STRING=your_azure_storage_connection_string
     ```

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Access the web interface at `http://localhost:5000`

3. Upload your OFW message log PDF (downloaded from the OFW platform)

4. Select an AI model and generate reports

## Configuration Options

Customize the analysis by modifying these settings in `app.py`:
- `MAX_INPUT_CHARS`: Maximum characters sent to AI models
- `MAX_RETRIES`: Number of retry attempts for API calls
- `AZURE_OAI_MAX_TOKENS`: Maximum tokens for Azure OpenAI responses
- `MAX_WORKERS`: Concurrent processing threads

## Requirements

- Python 3.8+
- See [requirements.txt](requirements.txt) for full dependency list

## Limitations

- Currently only processes English text
- PDF parsing depends on OFW's consistent export format
- AI analysis quality depends on model capabilities and prompt engineering

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is provided for informational purposes only. The analysis results should not be considered legal advice or used as evidence without professional review. The developers are not responsible for any decisions made based on this tool's output.
