# -*- coding: utf-8 -*-
"""
Flask application for parsing OurFamilyWizard message logs (PDF format),
analyzing message content using AI (Google Gemini or Azure OpenAI),
generating monthly and yearly summary reports (PDF format), and uploading
results to Azure Blob Storage. Provides a web interface for file upload
and report generation triggering.
"""

import os
import io
import re
import fitz  # PyMuPDF for PDF processing
import uuid
import json # For saving/loading intermediate data
import requests # For Azure OpenAI API calls
import time # For delays in retry logic
import shutil # For directory operations (cleanup)
import zipfile # For creating ZIP archives of reports
from datetime import datetime
from collections import defaultdict
import concurrent.futures # For parallel processing of tasks (uploads, AI calls)
import threading # Potentially for thread-local data (though not explicitly used here)

# --- Google AI Imports ---
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError

# --- Flask Imports ---
from flask import Flask, request, render_template, flash, redirect, url_for, send_file, session

# --- ReportLab Imports (for PDF Generation) ---
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib import colors

# --- Azure SDK Imports ---
from azure.storage.blob import BlobServiceClient, ContainerClient

# --- Configuration ---
# Azure Blob Storage credentials and container name
AZURE_STORAGE_CONNECTION_STRING = "InsertYourConnectionString"
AZURE_STORAGE_CONTAINER_NAME = "tempfiles"

# Google AI API Key and Model
GOOGLE_API_KEY = "InsertYourAPIKey"
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Model for Google AI analysis

# Azure OpenAI Service configuration
AZURE_OAI_ENDPOINT = "YourEndPoint"
AZURE_OAI_KEY = "YourAPIKey"
AZURE_OAI_MAX_TOKENS = 16000 # Max tokens for Azure OpenAI response

# AI Call Settings
MAX_RETRIES = 3 # Maximum number of retries for failed AI API calls
INITIAL_WAIT_SECONDS = 10 # Initial wait time before retrying API call
WAIT_INCREMENT_SECONDS = 15 # Additional seconds to wait for each subsequent retry
MAX_INPUT_CHARS = 450000 # Maximum characters allowed as input to the AI model (to prevent errors/truncation)
# Number of parallel workers for concurrent tasks (uploads, AI calls).
# Reduced significantly due to observed rate limiting issues with AI APIs.
# Adjust based on API limits and server resources.
MAX_WORKERS = 3

# --- Temporary Data Storage ---
# Directory to store temporary session data (parsed messages) between steps
TEMP_DATA_DIR = os.path.join(os.path.dirname(__file__), 'temp_session_data')
os.makedirs(TEMP_DATA_DIR, exist_ok=True)

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # Required for session management (flash messages, storing process state)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax' # Session cookie security setting

# --- Configure Google AI Client ---
# Initialize the Google AI client upon application start
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Google AI SDK configured successfully.")
except Exception as e:
    # Log a warning if configuration fails, Gemini model won't be usable
    print(f"WARNING: Failed to configure Google AI SDK: {e}. Check API Key if intending to use Gemini.")

# --- Shared Prompt Generation ---
# Functions to create the standardized prompts for the AI models

def create_analysis_prompt(participants, month_year_str, combined_messages):
    """Generates the standardized prompt for MONTHLY analysis for either AI model."""
    # Prompt asks the AI to analyze monthly messages and return specific data points
    # in a structured JSON format.
    return f"""
Analyze the following messages exchanged between {participants}. The messages cover the period of {month_year_str}.

Provide the analysis STRICTLY in the following JSON format. Do not include any text before or after the JSON object. Ensure the JSON is valid:
{{
  "summary": "Provide a concise summary of the key discussion points, decisions, and events mentioned in the messages for {month_year_str}.",
  "drug_mentions": [],
  "alcohol_mentions": [],
  "money_mentions": [],
  "conflict_negativity_analysis": "Describe the overall sentiment and tone for {month_year_str}. Highlight specific examples (quotes if possible) of conflict, negativity, anger, or frustration. If the tone is generally neutral or positive, state that."
}}

Instructions for Monthly Analysis:
- For "summary": Be objective and focus on the main topics discussed during this specific month.
- For the lists "drug_mentions", "alcohol_mentions", "money_mentions":
    - The value for these keys MUST be a JSON list containing JSON objects.
    - Each object in the list MUST have exactly THREE keys: "timestamp" (string), "sender" (string), and "quote" (string).
    - Example of a correctly formatted list with MULTIPLE items:
      "money_mentions": [
        {{"timestamp": "MM/DD/YYYY at HH:MM AM/PM", "sender": "Sender Name", "quote": "Quote mentioning money."}},
        {{"timestamp": "MM/DD/YYYY at HH:MM AM/PM", "sender": "Another Person", "quote": "Another quote about finances."}}
      ]
    - Search for explicit references within this month's messages. Include related terms (e.g., 'tuition', 'payment', 'debt', 'cost', 'HOA', 'miles', 'bank', 'insurance', 'ATM', 'dollars', '$', 'reimburse' for money; 'beer', 'wine', 'drinking' for alcohol; 'marijuana', 'pills', 'substance', 'herbs' for drugs).
    - If mentions are found, extract the EXACT 'Sent: MM/DD/YYYY at HH:MM AM/PM' timestamp from the specific message where the quote appears.
    - IMPORTANT: Also extract the sender's name (the value usually found after 'From:' near the start of the message) for the **sender** key.
    - Provide the most relevant quote containing the mention for the **quote** key.
    - If NO mentions are found for a category, return an empty list for that key (e.g., "drug_mentions": []). Do NOT put strings like "None found" inside the list.
- For "conflict_negativity_analysis": Assess the emotional tone specific to this month. Use quotes to support the analysis where appropriate. Be specific about the nature of the negativity.

Messages Log for {month_year_str}:
{combined_messages}
"""

def create_yearly_analysis_prompt(participants, year, formatted_monthly_analyses):
    """Generates the prompt for YEARLY analysis synthesis."""
    # Prompt asks the AI to synthesize previously generated monthly analyses
    # for a given year into a consolidated yearly summary, using a similar JSON structure.
    return f"""
Synthesize the provided monthly analyses for the entire year {year} involving participants {participants}.

Your goal is to create a single, consolidated YEARLY analysis. Provide this yearly analysis STRICTLY in the following JSON format. Do not include any text before or after the JSON object. Ensure the JSON is valid:
{{
  "summary": "Provide a concise overarching summary of the key discussion themes, major decisions, significant events, and overall communication patterns observed throughout the entire year {year}.",
  "drug_mentions": [],
  "alcohol_mentions": [],
  "money_mentions": [],
  "conflict_negativity_analysis": "Describe the dominant sentiment and tone across the whole year {year}. Highlight recurring conflicts, significant negative interactions, or overall trends in negativity/positivity observed across the months. Mention if the tone shifted significantly during the year."
}}

Instructions for Yearly Synthesis:
- Review all the monthly data provided below.
- For "summary": Create a high-level summary covering the entire year. Do not just list monthly summaries; synthesize the key points and trends.
- For the lists "drug_mentions", "alcohol_mentions", "money_mentions":
    - Aggregate the most significant or recurring mentions from the *monthly* reports. You do not need to include *every single* mention from the entire year if there are many duplicates or minor instances. Prioritize clarity and significance for the yearly overview.
    - The format MUST remain a JSON list of objects, each with "timestamp", "sender", and "quote". Ensure you include the sender's name associated with each aggregated quote.
    - If NO significant mentions were noted across the year in the monthly reports for a category, return an empty list (e.g., "drug_mentions": []).
- For "conflict_negativity_analysis": Provide a holistic view of the conflict and tone for the year. Identify major arguments, persistent issues, or overall relationship dynamics evident from the monthly analyses. Note any significant escalations or de-escalations.

Monthly Analysis Data for {year}:
{formatted_monthly_analyses}
"""

# --- Helper Functions ---
# Utility functions for PDF parsing, text cleaning, and basic PDF creation.

def extract_text_and_participants(pdf_stream):
    """
    Extracts text content and attempts to find participant names from a PDF file stream.

    Args:
        pdf_stream: A file-like object containing the PDF data.

    Returns:
        A tuple containing:
        - full_text (str): The concatenated text content of all pages.
        - participants (str): A comma-separated string of detected participant names,
                              or "Unknown" if not found.
    Raises:
        Exception: If there's an error opening or reading the PDF using PyMuPDF.
    """
    full_text = ""
    participants = "Unknown"
    try:
        # Open PDF from memory stream
        with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
            # Attempt to find participants in the first few pages
            header_text = ""
            for i in range(min(10, len(doc))): # Check first 10 pages
                 header_text += doc[i].get_text("text")
            # Regex to find the "Parents:" line
            participants_match = re.search(r'Parents:(.*?)\n', header_text, re.IGNORECASE | re.DOTALL)
            if participants_match:
                participants_raw = participants_match.group(1).strip()
                # Split and clean names
                participants_list = [p.strip() for p in re.split(r'[,\n]+', participants_raw) if p.strip()]
                participants = ', '.join(participants_list)
            else:
                 # Log warning if participants aren't found in the expected location
                 print("Warning: Could not find 'Parents:' line in the first few pages.", flush=True)

            # Extract text from all pages
            total_pages = len(doc)
            print(f"Reading {total_pages} pages from PDF...")
            page_texts = [page.get_text("text") for page in doc]
            # Join pages with form feed character (helps in potential splitting later)
            full_text = "\f".join(page_texts)
            print(f"  Finished reading {total_pages} pages.")
    except Exception as e:
        # Log and re-raise errors during PDF processing
        print(f"Error opening or reading PDF: {e}", flush=True)
        raise
    return full_text, participants

def clean_message_text(text):
    """
    Removes common headers, footers, and extra whitespace from the extracted PDF text.

    Args:
        text (str): The raw text extracted from the PDF.

    Returns:
        str: The cleaned text content.
    """
    # Remove page numbers (e.g., "Page 1 of 10") possibly followed by form feed
    text = re.sub(r'Page \d+ of \d+\s*\f?', '', text, flags=re.MULTILINE)
    # Remove OurFamilyWizard headers/footers
    text = re.sub(r'OurFamilyWizard\s+ourfamilywizard\.com.*?\n', '', text, flags=re.IGNORECASE)
    # Remove common metadata lines
    text = re.sub(r'\s*Generated:.*?\n', '', text)
    text = re.sub(r'\s*Number of messages:.*?\n', '', text)
    text = re.sub(r'\s*Timezone:.*?\n', '', text)
    text = re.sub(r'\s*Child\(ren\):.*?\n', '', text)
    text = re.sub(r'\s*Third Party:.*?\n', '', text)
    # Remove lines containing only whitespace or specific unicode space chars
    text = re.sub(r'^\s*\u2003\s*$', '', text, flags=re.MULTILINE)
    # Strip leading/trailing whitespace
    text = text.strip()
    # Normalize line endings
    text = re.sub(r'\r\n', '\n', text)
    # Collapse multiple blank lines into a single blank line
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def parse_messages(full_text):
    """
    Parses the cleaned full text into individual messages, grouping them by year and month.

    Args:
        full_text (str): The cleaned text containing all messages.

    Returns:
        defaultdict: A nested dictionary where keys are years (int),
                     values are dictionaries mapping months (int) to lists of message strings.
                     e.g., {2023: {1: ["message1", "message2"], 2: ["message3"]}}
    """
    # Split the text based on the "Message X of Y" header pattern
    message_chunks = re.split(r'(\s*Message \d+ of \d+\s*\n)', full_text)
    messages_by_year_month = defaultdict(lambda: defaultdict(list))
    total_messages_found = 0
    unparsed_chunks = []
    print("Parsing extracted text into individual messages...")
    num_chunks_to_process = len(message_chunks) // 2 # Each message has a header and content part

    # Iterate through chunks, pairing header and content
    processed_chunk_count = 0
    for i in range(1, len(message_chunks), 2):
        header = message_chunks[i]
        content = message_chunks[i+1] if (i + 1) < len(message_chunks) else ""
        full_message_text = (header + content).strip()
        processed_chunk_count += 1

        if not full_message_text: continue # Skip empty chunks

        # Find the 'Sent:' date within the message text
        sent_match = re.search(r'Sent:\s*(\d{1,2}/\d{1,2}/\d{4})\s+at\s+\d{1,2}:\d{2}\s+[AP]M', full_message_text, re.IGNORECASE | re.MULTILINE)
        if sent_match:
            date_str = sent_match.group(1)
            try:
                # Parse the date to get year and month
                sent_date = datetime.strptime(date_str, '%m/%d/%Y')
                year = sent_date.year
                month = sent_date.month
                # Add the full message text to the appropriate year/month bucket
                messages_by_year_month[year][month].append(full_message_text)
                total_messages_found += 1
            except ValueError:
                # Log warnings for messages where the date couldn't be parsed
                print(f"Warning: Could not parse date '{date_str}' in message chunk {processed_chunk_count}/{num_chunks_to_process}. Header: {header.strip()[:100]}...", flush=True)
                unparsed_chunks.append(full_message_text)
        else:
            # If no 'Sent:' date found, treat as unparsed
            unparsed_chunks.append(full_message_text)

        # Log progress periodically
        if processed_chunk_count % 100 == 0:
            print(f"  Parsed {processed_chunk_count}/{num_chunks_to_process} potential message chunks...", flush=True)

    print(f"Finished parsing. Found {total_messages_found} messages with valid dates.", flush=True)
    if unparsed_chunks:
        print(f"Could not parse dates for {len(unparsed_chunks)} chunks (see warnings above if any). These messages will be excluded from analysis.", flush=True)

    return messages_by_year_month

def create_monthly_pdf(messages, year, month):
    """
    Creates a simple PDF document containing the raw messages for a specific month.

    Args:
        messages (list): A list of message strings for the given month.
        year (int): The year.
        month (int): The month.

    Returns:
        io.BytesIO: A memory buffer containing the generated PDF data.

    Raises:
        Exception: If there's an error building the PDF using ReportLab.
    """
    pdf_buffer = io.BytesIO()
    # Setup ReportLab document template
    doc = SimpleDocTemplate(pdf_buffer, pagesize=fitz.paper_size("letter"),
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    # Define basic styles for the raw message log
    styles['Normal'].fontName = 'Courier' # Monospaced font for message text
    styles['Normal'].fontSize = 9
    styles['Normal'].leading = 11
    styles['BodyText'].alignment = TA_LEFT
    styles.add(ParagraphStyle(name='CenteredTitle', parent=styles['h1'], alignment=TA_CENTER))

    story = [] # List of flowables for ReportLab

    # Add title
    title = f"Raw Messages Log: {datetime(year, month, 1).strftime('%B %Y')}"
    story.append(Paragraph(title, styles['CenteredTitle']))
    story.append(Spacer(1, 0.2*inch))

    # Add each message, formatted for PDF display
    for i, msg_text in enumerate(messages):
        # Basic cleanup for display (remove page numbers, convert newlines)
        cleaned_display_text = re.sub(r'Page \d+ of \d+\s*\f?', '', msg_text, flags=re.MULTILINE).strip()
        cleaned_display_text = cleaned_display_text.replace('\n', '<br/>\n') # Convert newlines to <br> for ReportLab
        message_block = Paragraph(cleaned_display_text, styles['Normal'])

        # Add a separator between messages, kept together with the preceding message
        if i < len(messages) - 1:
            story.append(KeepTogether([message_block, Spacer(1, 0.05*inch), Paragraph("---", styles['Normal']), Spacer(1, 0.10*inch)]))
        else:
            # Just add the last message block
            story.append(message_block)

    try:
        # Build the PDF document in memory
        doc.build(story)
        pdf_buffer.seek(0) # Rewind the buffer for reading
        return pdf_buffer
    except Exception as e:
        # Log details if PDF generation fails
        print(f"Error building raw message PDF for {year}-{month:02d}: {e}", flush=True)
        first_message_snippet = messages[0][:500] if messages else "N/A"
        print(f"Problematic text snippet (first 500 chars of first message): {first_message_snippet}", flush=True)
        raise # Re-raise the exception


# --- Azure OpenAI API Call Function ---

def call_azure_openai(input_content, participants, time_period_str, prompt_creator_func):
    """
    Calls the Azure OpenAI API to perform analysis, with built-in retry logic for common errors.

    Args:
        input_content (str): The text content (messages or monthly summaries) to analyze.
        participants (str): The names of the participants involved.
        time_period_str (str): A string describing the time period (e.g., "May 2023", "2023").
        prompt_creator_func (callable): Function to generate the specific prompt
                                       (e.g., create_analysis_prompt or create_yearly_analysis_prompt).

    Returns:
        dict: A dictionary containing the parsed JSON analysis from the API,
              or an error dictionary if the call fails or the response is invalid.
              Example error: {"error": "API Rate Limit Exceeded", "details": "..."}
    """
    REQ_TIMEOUT = 400 # Request timeout in seconds
    print(f"Preparing Azure OpenAI call for {time_period_str}...", flush=True)

    # Generate the prompt using the provided function
    prompt = prompt_creator_func(participants, time_period_str, input_content)

    # Prepare request headers and payload
    headers = {"Content-Type": "application/json", "api-key": AZURE_OAI_KEY}
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": AZURE_OAI_MAX_TOKENS,
        "temperature": 0.2, # Lower temperature for more deterministic output
        "top_p": 0.95,
        "response_format": {"type": "json_object"} # Request JSON output
    }

    current_wait = INITIAL_WAIT_SECONDS # Initial delay for retries
    # Retry loop
    for attempt in range(MAX_RETRIES + 1):
        print(f"  Attempt {attempt + 1}/{MAX_RETRIES + 1} calling Azure OpenAI for {time_period_str}...", flush=True)
        try:
            # Make the POST request
            response = requests.post(AZURE_OAI_ENDPOINT, headers=headers, json=payload, timeout=REQ_TIMEOUT)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # Process successful response
            response_json = response.json()
            if "choices" in response_json and len(response_json["choices"]) > 0:
                message_content = response_json["choices"][0].get("message", {}).get("content")
                if message_content:
                    try:
                        # Clean up potential markdown/code blocks around the JSON
                        cleaned_content = message_content.strip().strip('```json').strip('```').strip()
                        # Parse the JSON response
                        analysis_data = json.loads(cleaned_content)
                        print(f"  Successfully received and parsed analysis from Azure OpenAI for {time_period_str}.", flush=True)
                        return analysis_data # Success!
                    except json.JSONDecodeError as json_err:
                        # Handle cases where the response is not valid JSON
                        print(f"Error: Could not decode JSON response from Azure OpenAI for {time_period_str}. Error: {json_err}", flush=True)
                        print(f"Raw content received:\n{message_content}", flush=True)
                        return {"error": "Failed to parse Azure OpenAI response JSON", "details": f"JSONDecodeError: {json_err}", "raw_content": message_content}
                else:
                    # Handle cases where the response structure is unexpected (missing content)
                    print(f"Error: 'content' missing in Azure OpenAI response message for {time_period_str}.", flush=True)
                    return {"error": "Azure OpenAI response content was empty"}
            else:
                # Handle cases where the response structure is unexpected (missing choices)
                print(f"Error: Unexpected Azure OpenAI response structure for {time_period_str}: {response_json}", flush=True)
                return {"error": "Unexpected Azure OpenAI response structure"}

        except requests.exceptions.Timeout:
            # Handle request timeouts
            print(f"Error: Timeout during Azure OpenAI call attempt {attempt + 1} for {time_period_str} after {REQ_TIMEOUT}s.", flush=True)
            if attempt < MAX_RETRIES:
                print(f"  Waiting {current_wait} seconds before retry...", flush=True)
                time.sleep(current_wait)
                current_wait += WAIT_INCREMENT_SECONDS # Increase wait time for next retry
                continue # Go to the next attempt
            else:
                 # Max retries exceeded for timeout
                 return {"error": f"API Request Timed Out after {MAX_RETRIES + 1} attempts", "details": f"Timeout was {REQ_TIMEOUT}s"}

        except requests.exceptions.RequestException as e:
            # Handle other request-related errors (network issues, HTTP errors)
            print(f"Error during Azure OpenAI call attempt {attempt + 1}: {e}", flush=True)
            # Check specifically for 429 Rate Limit error
            is_429_error = hasattr(e, 'response') and e.response is not None and e.response.status_code == 429
            if is_429_error and attempt < MAX_RETRIES:
                # Handle rate limiting with retry logic
                retry_after = e.response.headers.get("Retry-After") # Check for Retry-After header
                wait_time = current_wait
                if retry_after:
                    try:
                        wait_time = int(retry_after) + 1 # Use header value if available
                        print(f"  Received 429 Too Many Requests. Using Retry-After header: waiting {wait_time} seconds...", flush=True)
                    except ValueError:
                        print(f"  Received 429 Too Many Requests. Could not parse Retry-After header ('{retry_after}'). Waiting default {wait_time} seconds...", flush=True)
                        current_wait += WAIT_INCREMENT_SECONDS # Increase default wait if header invalid
                else:
                     # No Retry-After header, use the increasing default wait
                     print(f"  Received 429 Too Many Requests. Waiting {wait_time} seconds before retry...", flush=True)
                     current_wait += WAIT_INCREMENT_SECONDS
                time.sleep(wait_time)
                continue # Go to the next attempt
            elif is_429_error and attempt == MAX_RETRIES:
                 # Max retries exceeded for rate limit error
                 print(f"  Max retries ({MAX_RETRIES}) exceeded for 429 error.", flush=True)
                 return {"error": f"API Rate Limit Exceeded after {MAX_RETRIES + 1} attempts", "details": str(e)}
            else:
                # Handle other non-retryable HTTP errors or network issues
                error_details = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    try: error_details += f" - Response Code: {e.response.status_code} - Response Body: {e.response.text}"
                    except Exception: pass # Ignore errors trying to get response details
                print(f"  Non-retryable HTTP error or network error occurred: {error_details}", flush=True)
                return {"error": f"API Request Failed", "details": error_details}

        except Exception as e:
             # Catch any other unexpected errors during the process
             print(f"  An unexpected error occurred during Azure OpenAI call attempt {attempt + 1}: {e}", flush=True)
             import traceback
             traceback.print_exc() # Print stack trace for debugging
             return {"error": f"Unexpected error during API call processing", "details": str(e)}

    # Fallback if all retries fail
    print(f"Azure OpenAI call failed for {time_period_str} after all attempts (fallback exit).", flush=True)
    return {"error": f"API call failed after {MAX_RETRIES + 1} attempts."}


# --- Google Gemini API Call Function ---

def call_gemini_flash(input_content, participants, time_period_str, prompt_creator_func):
    """
    Calls the Google Gemini API to perform analysis, with built-in retry logic for rate limits.

    Args:
        input_content (str): The text content (messages or monthly summaries) to analyze.
        participants (str): The names of the participants involved.
        time_period_str (str): A string describing the time period (e.g., "May 2023", "2023").
        prompt_creator_func (callable): Function to generate the specific prompt
                                       (e.g., create_analysis_prompt or create_yearly_analysis_prompt).

    Returns:
        dict: A dictionary containing the parsed JSON analysis from the API,
              or an error dictionary if the call fails or the response is invalid/blocked.
              Example error: {"error": "API Rate Limit Exceeded", "details": "..."}
    """
    print(f"Preparing Gemini call for {time_period_str}...", flush=True)

    # Generate the prompt
    prompt = prompt_creator_func(participants, time_period_str, input_content)

    # Configure safety settings to block only high-risk content
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
    # Configure generation parameters (token limit, temperature, JSON output)
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=8192, # Gemini Flash has a large context window
        temperature=0.2,
        top_p=0.95,
        response_mime_type="application/json" # Request JSON output directly
    )
    # Initialize the generative model
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        safety_settings=safety_settings,
        generation_config=generation_config
    )

    current_wait = INITIAL_WAIT_SECONDS # Initial delay for retries
    # Retry loop
    for attempt in range(MAX_RETRIES + 1):
        print(f"  Attempt {attempt + 1}/{MAX_RETRIES + 1} calling Google Gemini for {time_period_str}...", flush=True)
        try:
            # Make the API call
            response = model.generate_content(prompt, request_options={'timeout': 400}) # Set timeout

            raw_text = "N/A" # Placeholder for raw response text

            # --- Process Gemini Response ---
            try:
                # Check for safety blocks first
                if not response.candidates and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason.name
                    block_details = response.prompt_feedback.safety_ratings if response.prompt_feedback.safety_ratings else "No specific ratings."
                    print(f"Error: Gemini response BLOCKED for {time_period_str}. Reason: {block_reason}. Details: {block_details}", flush=True)
                    return {"error": f"Gemini response blocked due to safety settings", "details": f"Reason: {block_reason}"}

                # Check if response has candidates (actual generated content)
                elif not response.candidates:
                    response_parts = getattr(response, 'parts', 'N/A') # Try to get parts for debugging
                    print(f"Error: Gemini response received but has no candidates for {time_period_str}. Response parts: {response_parts}", flush=True)
                    return {"error": "Gemini response contained no valid candidates", "details": f"Response parts: {response_parts}"}

                # If candidates exist, get the text content
                raw_text = response.text
                cleaned_text = raw_text.strip() # Basic cleaning

                # Parse the JSON response
                analysis_data = json.loads(cleaned_text)
                print(f"  Successfully received and parsed analysis from Google Gemini for {time_period_str}.", flush=True)
                return analysis_data # Success!

            except (json.JSONDecodeError) as json_err:
                # Handle JSON parsing errors
                print(f"Error: Could not decode JSON response from Gemini for {time_period_str}. Error: {json_err}", flush=True)
                print(f"Raw text received that failed parsing:\n{raw_text}", flush=True)
                # Check if a block reason might be related (content altered/removed)
                block_reason = "None"
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                     block_reason = response.prompt_feedback.block_reason.name
                     print(f"-> Gemini content might have been blocked/altered leading to JSON error. Reason: {block_reason}")
                     return {"error": f"Failed to parse Gemini JSON (potentially blocked/altered: {block_reason})", "details": f"JSONDecodeError: {json_err}", "raw_content": raw_text}
                else:
                    # Return standard JSON error if no block reason found
                    return {"error": "Failed to parse Gemini response JSON", "details": f"JSONDecodeError: {json_err}", "raw_content": raw_text}

            except (AttributeError, TypeError, ValueError) as attr_err:
                 # Handle unexpected response structure or value errors
                 print(f"Error: Unexpected response structure or value from Gemini for {time_period_str}. Error: {attr_err}", flush=True)
                 print(f"Raw Response Object (if available): {response}", flush=True)
                 return {"error": "Unexpected response structure or value from Gemini", "details": str(attr_err)}

        except ResourceExhausted as e:
            # Handle rate limit errors (ResourceExhausted)
            print(f"Error during Gemini call attempt {attempt + 1}: {e}", flush=True)
            if attempt < MAX_RETRIES:
                print(f"  Received ResourceExhausted (Rate Limit). Waiting {current_wait} seconds before retry...", flush=True)
                time.sleep(current_wait)
                current_wait += WAIT_INCREMENT_SECONDS # Increase wait time
                continue # Go to next attempt
            else:
                # Max retries exceeded for rate limit
                print(f"  Max retries ({MAX_RETRIES}) exceeded for rate limit error.", flush=True)
                return {"error": f"API Rate Limit Exceeded after {MAX_RETRIES + 1} attempts", "details": str(e)}

        except GoogleAPIError as e:
             # Handle other non-retryable Google API errors
             print(f"  Non-retryable Google API error occurred during attempt {attempt + 1}: {e}", flush=True)
             return {"error": f"Google API Call Failed", "details": str(e)}

        except Exception as e:
             # Catch any other unexpected errors
             print(f"  An unexpected error occurred during Gemini call attempt {attempt + 1}: {e}", flush=True)
             import traceback
             traceback.print_exc() # Print stack trace for debugging
             return {"error": f"Unexpected error during API call processing", "details": str(e)}

    # Fallback if all retries fail
    print(f"Gemini call failed for {time_period_str} after all attempts (fallback exit).", flush=True)
    return {"error": f"API call failed after {MAX_RETRIES + 1} attempts."}


# --- PDF Analysis Report Creation ---

def create_analysis_report_pdf(participants, time_period_str, analysis_data, report_type="Monthly",
                               temp_report_dir=None, master_folder_name=None, selected_model=None, year=None, month=None):
    """
    Creates the analysis report PDF (monthly or yearly) using ReportLab based on AI analysis data.
    Can generate either a standard report or an error report if analysis failed.
    Optionally saves the generated PDF to a local temporary directory.

    Args:
        participants (str): Names of participants.
        time_period_str (str): Description of the period (e.g., "May 2023", "2023").
        analysis_data (dict): The dictionary returned by the AI call function
                              (contains analysis or error info).
        report_type (str): "Monthly" or "Yearly".
        temp_report_dir (str, optional): Base directory for saving temporary reports.
        master_folder_name (str, optional): Unique folder name for this processing run.
        selected_model (str, optional): Name of the AI model used ('azure' or 'gemini').
        year (int, optional): The year (required for constructing save path).
        month (int, optional): The month (required for monthly report save path).

    Returns:
        str | None: The absolute path to the saved local PDF file if saving was requested
                    and successful, otherwise None.
    """
    pdf_buffer = io.BytesIO()
    # Setup ReportLab document template
    doc = SimpleDocTemplate(pdf_buffer, pagesize=fitz.paper_size("letter"),
                            leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()

    # Define custom styles for the report
    styles.add(ParagraphStyle(name='SectionHeading', parent=styles['h2'], spaceBefore=12, spaceAfter=6, fontSize=12))
    styles.add(ParagraphStyle(name='SubHeading', parent=styles['h3'], spaceBefore=8, spaceAfter=4, fontSize=10, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name='NormalRight', parent=styles['Normal'], alignment=TA_LEFT)) # Note: Alignment seems LEFT despite name
    styles.add(ParagraphStyle(name='ListItemStyle', parent=styles['Normal'], leftIndent=18))
    # Style for displaying extracted quotes
    styles.add(ParagraphStyle(name='QuoteStyle', parent=styles['Normal'], leftIndent=24, rightIndent=18, fontName='Courier', fontSize=9, textColor=colors.dimgrey, spaceAfter=4, borderPadding=(2, 2, 2, 6), borderColor=colors.lightgrey, borderLeftWidth=1))
    styles.add(ParagraphStyle(name='ErrorStyle', parent=styles['Normal'], textColor=colors.red))
    styles.add(ParagraphStyle(name='CenteredTitle', parent=styles['h1'], alignment=TA_CENTER))

    story = [] # List of flowables for ReportLab

    # --- Report Header ---
    story.append(Paragraph(f"OurFamilyWizard {report_type} Message Analysis", styles['CenteredTitle']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%m/%d/%Y %I:%M %p')}", styles['Normal']))
    if selected_model:
        model_name_display = "Google Gemini" if selected_model == 'gemini' else "Azure OpenAI"
        story.append(Paragraph(f"AI Model Used: {model_name_display}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # --- Determine if Analysis Failed ---
    analysis_failed = analysis_data.get("error") is not None
    error_suffix = "_ERROR" if analysis_failed else "" # Suffix for error report filenames

    # --- Determine Local Save Path (if requested) ---
    report_filename = None
    target_dir = None
    local_pdf_path = None
    if temp_report_dir and master_folder_name and selected_model and year:
        # Construct path based on report type (Monthly/Yearly)
        if report_type == "Monthly" and month:
            report_filename = f"analysis_report_{year}_{month:02d}{error_suffix}.pdf"
            # Structure: /temp_dir/master_folder/reports/model/year/month/report.pdf
            target_dir = os.path.join(temp_report_dir, master_folder_name, "reports", selected_model, str(year), f"{month:02d}")
        elif report_type == "Yearly":
            report_filename = f"yearly_analysis_report_{year}{error_suffix}.pdf"
            # Structure: /temp_dir/master_folder/reports/model/year/report.pdf
            target_dir = os.path.join(temp_report_dir, master_folder_name, "reports", selected_model, str(year))

        if target_dir and report_filename:
            local_pdf_path = os.path.join(target_dir, report_filename)
            try:
                # Ensure the target directory exists
                os.makedirs(target_dir, exist_ok=True)
            except OSError as e:
                 # Log error if directory creation fails, disable saving
                 print(f"!! Error creating directory {target_dir}: {e}", flush=True)
                 local_pdf_path = None # Cannot save if directory fails

    # --- Generate Error Report PDF (if analysis failed) ---
    if analysis_failed:
        story.append(Paragraph(f"Error During {report_type} Analysis", styles['SectionHeading']))
        story.append(Paragraph(f"Could not generate full analysis for {time_period_str}.", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"<b>Error:</b> {analysis_data.get('error', 'Unknown Error')}", styles['ErrorStyle']))
        # Include error details if available
        if "details" in analysis_data and analysis_data['details']:
            # Ensure details are safely encoded/decoded for ReportLab compatibility
            details_text = str(analysis_data['details']).encode('latin-1', 'replace').decode('latin-1')
            details_text = details_text.replace('\n', '<br/>\n') # Convert newlines
            story.append(Paragraph(f"<b>Details:</b> {details_text}", styles['Normal']))
        # Include raw AI response if available (useful for debugging JSON errors)
        if "raw_content" in analysis_data and analysis_data['raw_content']:
             story.append(Paragraph("Raw AI Response (if available):", styles['SubHeading']))
             raw_content_formatted = str(analysis_data['raw_content']).encode('latin-1', 'replace').decode('latin-1')
             raw_content_formatted = raw_content_formatted.replace('\n', '<br/>\n')
             story.append(Paragraph(raw_content_formatted, styles['QuoteStyle']))

        # Build the error report PDF
        try:
             doc.build(story)
        except Exception as build_err:
             # Handle errors during the error report build itself (less likely)
             print(f"Error building the error report PDF itself ({time_period_str}): {build_err}", flush=True)
             pdf_buffer = io.BytesIO() # Reset buffer
             # Create a very simple fallback error text
             error_text = f"Critical Error: Could not generate {report_type} report PDF for {time_period_str}.\nAnalysis Error: {analysis_data.get('error', 'Unknown')}\n"
             if "details" in analysis_data: error_text += f"Details: {analysis_data['details']}\n"
             error_text += f"PDF Build Error: {build_err}"
             # Try a minimal ReportLab build
             simple_doc = SimpleDocTemplate(pdf_buffer)
             simple_story = [Paragraph(line, styles['Normal']) for line in error_text.split('\n')]
             try: simple_doc.build(simple_story)
             except: pdf_buffer.write(error_text.encode('utf-8', 'replace')) # Absolute fallback

        pdf_buffer.seek(0) # Rewind buffer

        # Save the error PDF locally if path is valid
        if local_pdf_path:
            try:
                print(f"  Attempting to save {report_type} error PDF locally to: {local_pdf_path}", flush=True)
                with open(local_pdf_path, 'wb') as f_out:
                    shutil.copyfileobj(pdf_buffer, f_out) # Copy buffer contents to file
                print(f"  Successfully saved error PDF locally.", flush=True)
                return local_pdf_path # Return path to saved error PDF
            except Exception as save_err:
                 print(f"!! Warning: Failed to save {report_type} error PDF locally to {local_pdf_path}: {save_err}", flush=True)
        return None # Return None if saving wasn't requested or failed

    # --- Normal Report Generation (if analysis succeeded) ---
    # Add Title / Period Info
    if report_type == "Yearly":
        story.append(Paragraph(f"{time_period_str} – Full Year Analysis", styles['SectionHeading']))
    else: # Monthly Report
        story.append(Paragraph(f"Monthly Analysis: {time_period_str}", styles['SectionHeading']))
    story.append(Paragraph(f"<b>Participants:</b> {participants or 'Unknown'}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Add Summary Section
    story.append(Paragraph("Summary", styles['SubHeading']))
    summary_text = analysis_data.get('summary', 'Summary not available.')
    # Ensure text is safe for ReportLab (handle potential encoding issues)
    summary_text_safe = summary_text.encode('latin-1', 'replace').decode('latin-1')
    story.append(Paragraph(summary_text_safe.replace('\n', '<br/>\n'), styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Add Mention Sections (Drugs, Alcohol, Money)
    mention_sections = [
        ('Mentions of Drugs', 'drug_mentions'),
        ('Mentions of Alcohol', 'alcohol_mentions'),
        ('Mentions of Money / Finances', 'money_mentions'),
    ]
    for title, key in mention_sections:
        story.append(Paragraph(title, styles['SubHeading']))
        mentions = analysis_data.get(key, []) # Get list of mentions from analysis data

        # Check if mentions data is a valid list
        if mentions and isinstance(mentions, list):
            list_items = [] # ReportLab ListItems
            for mention in mentions:
                # Check if each mention item is a dictionary as expected
                if isinstance(mention, dict):
                    # Extract timestamp, quote, and sender
                    ts = mention.get('timestamp', 'Timestamp N/A')
                    qt = mention.get('quote', 'Quote N/A')
                    sender = mention.get('sender', 'Sender N/A')

                    # Clean and format the quote text for display
                    quote_text = str(qt) if qt is not None else 'Quote N/A'
                    # Basic HTML entity escaping and handling common problematic chars
                    quote_text_safe = quote_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    quote_text_safe = quote_text_safe.replace(chr(8217), "'").replace(chr(8220), '"').replace(chr(8221), '"')
                    quote_formatted = f"<i>“{quote_text_safe}”</i>" # Italicize and add quotes

                    # Create content for the list item (Sender/Timestamp + Formatted Quote)
                    item_content = [
                         Paragraph(f"<b>{sender} - {ts}:</b>", styles['ListItemStyle']),
                         Paragraph(quote_formatted, styles['QuoteStyle'])
                    ]
                    # Add as a ReportLab ListItem (bullet character used below)
                    list_items.append(ListItem(item_content, bulletFontSize=10, spaceAfter=8))
                else:
                     # Handle cases where AI returns malformed data in the list
                     malformed_item_text = f"<i>Invalid mention format received from AI: {str(mention)[:150]}...</i>"
                     list_items.append(ListItem(Paragraph(malformed_item_text, styles['ListItemStyle']), bulletFontSize=10))

            # If any valid list items were created, add the ListFlowable to the story
            if list_items:
                # Use the actual bullet character (u'\u2022')
                mention_list = ListFlowable(list_items, bulletType='bullet', start=u'\u2022', bulletFontSize=10, leftIndent=18)
                story.append(mention_list)
            else:
                 # Display message if list was empty after processing
                 story.append(Paragraph("No specific mentions identified in this period.", styles['Normal']))
        elif isinstance(mentions, str):
             # Handle case where AI returned a string instead of a list (unexpected)
             mentions_safe = mentions.encode('latin-1', 'replace').decode('latin-1')
             story.append(Paragraph(f"<i>AI provided text instead of a list: {mentions_safe}</i>", styles['Normal']))
        else:
            # Default message if no mentions found or data format incorrect
            story.append(Paragraph("No specific mentions found or data unavailable.", styles['Normal']))
        story.append(Spacer(1, 0.2*inch)) # Spacer after each mention section

    # Add Conflict/Negativity Analysis Section
    story.append(Paragraph("Conflict and Negativity Analysis", styles['SubHeading']))
    conflict_text = analysis_data.get('conflict_negativity_analysis', 'Analysis not available.')
    conflict_text_safe = conflict_text.encode('latin-1', 'replace').decode('latin-1') # Sanitize text
    story.append(Paragraph(conflict_text_safe.replace('\n', '<br/>\n'), styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # --- Build PDF Document ---
    try:
        print(f"  Building PDF document structure for {report_type} report: {time_period_str}...", flush=True)
        doc.build(story) # Build the PDF from the story flowables
        print(f"  Successfully built PDF for {time_period_str}.", flush=True)
        pdf_buffer.seek(0) # Rewind buffer

        # Save the generated PDF locally if path is valid
        if local_pdf_path:
            try:
                print(f"  Attempting to save {report_type} PDF locally to: {local_pdf_path}", flush=True)
                with open(local_pdf_path, 'wb') as f_out:
                    shutil.copyfileobj(pdf_buffer, f_out) # Write buffer to file
                print(f"  Successfully saved PDF locally.", flush=True)
                return local_pdf_path # Return path to saved report
            except Exception as save_err:
                 print(f"!! Warning: Failed to save {report_type} PDF locally to {local_pdf_path}: {save_err}", flush=True)
        return None # Return None if saving wasn't requested or failed

    except Exception as e:
        # --- Handle CRITICAL errors during PDF build (after successful analysis) ---
        print(f"!!! CRITICAL Error building {report_type} analysis report PDF for {time_period_str} (after successful analysis): {e}", flush=True)
        # Avoid complex error report if it's a specific known ReportLab issue (_FrameBreak)
        # which can sometimes happen with complex layouts, though efforts were made to avoid it.
        if not isinstance(e, NotImplementedError) or "_FrameBreak.wrap" not in str(e):
             import traceback
             traceback.print_exc() # Print traceback for unexpected build errors

        # --- Create a Simplified Error PDF indicating a build failure ---
        pdf_buffer = io.BytesIO() # Reset buffer
        error_doc = SimpleDocTemplate(pdf_buffer)
        # Simple content for the build error report
        error_story = [ Paragraph(f"{report_type} Analysis Report Generation Error for {time_period_str}", styles['h1']),
                        Paragraph("Analysis data was received successfully, but a critical error occurred during PDF document creation:", styles['Normal']),
                        Paragraph(f"<b>Error Type:</b> {e.__class__.__name__}", styles['ErrorStyle']),
                        Paragraph(f"<b>Error Details:</b> {str(e).encode('latin-1', 'replace').decode('latin-1')}", styles['ErrorStyle']),
                        Paragraph("Please check server logs for a full traceback.", styles['Normal']) ]
        try: error_doc.build(error_story) # Try to build the simple error report
        except Exception as final_build_err:
             # Absolute fallback: Write plain text to buffer if even simple build fails
             pdf_buffer = io.BytesIO() # Reset again
             error_text = f"Multiple errors generating report PDF for {time_period_str}.\nAnalysis OK.\nInitial Build Error: {e}\nFinal Build Error: {final_build_err}"
             pdf_buffer.write(error_text.encode('utf-8', 'replace'))

        pdf_buffer.seek(0) # Rewind buffer

        # Attempt to save the build error report locally
        if local_pdf_path:
            # Create a distinct filename for the build error report
            build_error_path = local_pdf_path.replace(".pdf", "_BUILD_ERROR.pdf")
            if build_error_path == local_pdf_path: build_error_path += "_BUILD_ERROR" # Ensure unique name
            try:
                print(f"  Attempting to save build-error PDF locally to: {build_error_path}", flush=True)
                with open(build_error_path, 'wb') as f_out:
                    shutil.copyfileobj(pdf_buffer, f_out)
                print(f"  Successfully saved build-error PDF locally.", flush=True)
                return build_error_path # Return path to the build error report
            except Exception as save_err:
                 print(f"!! Warning: Failed to save build-error PDF locally to {build_error_path}: {save_err}", flush=True)

        return None # Return None if couldn't save the build error report


# --- Azure Blob Upload Helper ---

def upload_to_azure(pdf_buffer_or_path, container_client, blob_name):
    """
    Uploads a BytesIO buffer or a local file path to Azure Blob Storage.

    Args:
        pdf_buffer_or_path (io.BytesIO | str): Either a BytesIO buffer containing the PDF data,
                                              or a string path to a local PDF file.
        container_client (ContainerClient): An initialized Azure ContainerClient instance.
        blob_name (str): The desired name (including path) for the blob in Azure.

    Raises:
        ValueError: If the input type is invalid or the file path doesn't exist.
        Exception: If the upload to Azure fails.
    """
    try:
        # Get a client for the specific blob
        blob_client = container_client.get_blob_client(blob_name)

        # Check if input is a buffer
        if isinstance(pdf_buffer_or_path, io.BytesIO):
            pdf_buffer_or_path.seek(0) # Ensure buffer is at the start
            blob_client.upload_blob(pdf_buffer_or_path, overwrite=True)
            print(f"    Successfully uploaded buffer to Azure: {blob_name}", flush=True)
        # Check if input is a valid file path
        elif isinstance(pdf_buffer_or_path, str) and os.path.exists(pdf_buffer_or_path):
            with open(pdf_buffer_or_path, "rb") as data: # Open file in binary read mode
                blob_client.upload_blob(data, overwrite=True)
            print(f"    Successfully uploaded file {os.path.basename(pdf_buffer_or_path)} to Azure: {blob_name}", flush=True)
        else:
             # Handle invalid input
             print(f"Error uploading to Azure: Invalid input type or file not found - {pdf_buffer_or_path}", flush=True)
             raise ValueError("Invalid input for upload_to_azure: must be BytesIO or existing file path")
    except Exception as e:
        # Log and re-raise errors during Azure upload
        print(f"Error uploading {blob_name} to Azure: {e}", flush=True)
        raise


# --- Yearly Report Generation Function ---

def generate_yearly_report_data_and_pdf(year, monthly_analyses_dict, participants, selected_model, master_folder_name, temp_report_dir):
    """
    Generates yearly analysis data by calling the appropriate AI model to synthesize
    monthly reports, then creates the yearly analysis report PDF locally.

    Args:
        year (int): The year for which to generate the report.
        monthly_analyses_dict (dict): Dictionary mapping month (int) to the successful
                                      monthly analysis data (dict) for that year.
        participants (str): Names of participants.
        selected_model (str): AI model identifier ('azure' or 'gemini').
        master_folder_name (str): Unique folder name for this processing run.
        temp_report_dir (str): Base directory for saving the temporary yearly report PDF.

    Returns:
        str | None: The path to the generated local yearly PDF (can be a success or error report),
                    or None if PDF generation/saving failed critically.
    """
    print(f"--- Preparing Yearly Analysis Data for {year} ({selected_model.upper()}) ---", flush=True)

    # Filter the input dictionary to ensure only valid, non-error monthly data is used
    valid_monthly_data_for_year = {}
    for month, analysis_data in monthly_analyses_dict.items():
        # Check if data exists and doesn't contain an "error" key
        if analysis_data and not analysis_data.get("error"):
            valid_monthly_data_for_year[month] = analysis_data
        # Optionally log excluded months (can be noisy if many errors occurred)
        # else:
        #     month_name = datetime(year, month, 1).strftime('%B')
        #     print(f"  Excluding {month_name} {year} from yearly analysis due to prior error or missing data.", flush=True)

    # If no valid monthly data exists for the year, generate an error report directly
    if not valid_monthly_data_for_year:
        print(f"Error: No successful monthly analysis data available for yearly report for {year}.", flush=True)
        error_data = {"error": f"Cannot generate yearly report for {year}", "details": "No successful monthly analyses were available for aggregation."}
        # Call the PDF creation function with the error data
        error_pdf_path = create_analysis_report_pdf(
            participants, str(year), error_data, report_type="Yearly",
            temp_report_dir=temp_report_dir, master_folder_name=master_folder_name,
            selected_model=selected_model, year=year
        )
        return error_pdf_path # Return path to the generated error PDF

    # --- Prepare Input for Yearly AI Call ---
    # Concatenate the JSON strings of valid monthly analyses
    input_for_ai = f"Monthly analyses for {year}:\n\n"
    sorted_months = sorted(valid_monthly_data_for_year.keys())
    for month in sorted_months:
        analysis = valid_monthly_data_for_year[month]
        month_name = datetime(year, month, 1).strftime('%B')
        try:
            # Serialize monthly data compactly for AI input
            analysis_str = json.dumps(analysis, separators=(',', ':'))
        except Exception as json_e:
            # Skip months that fail serialization (should be rare)
            print(f"Warning: Could not serialize valid analysis for {month_name} {year} to JSON: {json_e}. Skipping.", flush=True)
            continue
        input_for_ai += f"--- {month_name} {year} ---\n{analysis_str}\n\n"

    # --- Truncate Input if Exceeds Max Length ---
    original_length = len(input_for_ai)
    truncated = False
    if original_length > MAX_INPUT_CHARS:
        print(f"Warning: Truncating yearly input data for {year} from {original_length} to {MAX_INPUT_CHARS} chars.", flush=True)
        input_for_ai = input_for_ai[:MAX_INPUT_CHARS]
        # Try to truncate cleanly at the end of a JSON block
        last_newline = input_for_ai.rfind('\n')
        if last_newline != -1: input_for_ai = input_for_ai[:last_newline]
        input_for_ai += "\n...[TRUNCATED]" # Indicate truncation
        truncated = True

    # --- Call the Selected AI Model for Yearly Synthesis ---
    yearly_analysis_data = None
    year_str = str(year)
    if selected_model == 'azure':
        yearly_analysis_data = call_azure_openai(input_for_ai, participants, year_str, create_yearly_analysis_prompt)
    else: # gemini
        yearly_analysis_data = call_gemini_flash(input_for_ai, participants, year_str, create_yearly_analysis_prompt)

    # --- Generate the Yearly PDF Report (or Error Report if AI failed) ---
    if yearly_analysis_data.get("error"):
        print(f"!! Failed to get yearly analysis for {year} using {selected_model.upper()}: {yearly_analysis_data['error']}", flush=True)
        # Pass the error data received from the AI call to the PDF creator
        yearly_report_pdf_path = create_analysis_report_pdf(
            participants, year_str, yearly_analysis_data, report_type="Yearly",
            temp_report_dir=temp_report_dir, master_folder_name=master_folder_name,
            selected_model=selected_model, year=year
        )
    else:
         # Yearly analysis successful
         print(f"  Successfully obtained yearly analysis for {year} using {selected_model.upper()}.", flush=True)
         if truncated: print(f"  Note: Yearly analysis for {year} was generated based on TRUNCATED monthly data.", flush=True)
         # Pass the successful analysis data to the PDF creator
         yearly_report_pdf_path = create_analysis_report_pdf(
             participants, year_str, yearly_analysis_data, report_type="Yearly",
             temp_report_dir=temp_report_dir, master_folder_name=master_folder_name,
             selected_model=selected_model, year=year
         )

    # Check if PDF generation/saving itself failed
    if not yearly_report_pdf_path:
        print(f"!! CRITICAL Error: Failed to generate yearly report PDF for {year} even after AI call.", flush=True)
        return None # Indicate failure to create the PDF file

    return yearly_report_pdf_path # Return the path to the locally saved yearly PDF


# --- Flask Routes ---
# Define the web application endpoints and their logic.

@app.route('/', methods=['GET'])
def index():
    """Renders the main upload form (index.html)."""
    # Check if there's an active process ID in the session from a previous upload
    process_id = session.get('last_process_id')
    show_generate_button = False

    if process_id:
        # Retrieve data associated with the process ID
        session_data = session.get(process_id, {})
        temp_file_path = session_data.get('temp_file_path')
        # Check if the temporary data file still exists
        if temp_file_path and os.path.exists(temp_file_path):
            # If data exists, show the "Generate Reports" button
            show_generate_button = True
        else:
            # If temp file is missing (e.g., server restart, manual deletion),
            # clean up the stale session data.
            session.pop(process_id, None)
            if session.get('last_process_id') == process_id: session.pop('last_process_id', None)
            session.modified = True
            # Notify user if the file was expected but missing
            if temp_file_path:
                flash('Temporary data expired or missing. Please upload the PDF again.', 'info')

    # Retrieve and display any final status message from the previous report generation run
    final_status = session.pop('final_status', None)
    final_status_category = session.pop('final_status_category', 'info')
    if final_status:
        flash(final_status, final_status_category)

    # Render the main page template
    return render_template('index.html', show_generate_button=show_generate_button, process_id=process_id)

# --- Helper function for parallel raw log processing ---
def generate_and_upload_raw_log(year, month, messages, master_folder_name, container_client):
    """
    Generates the raw monthly message PDF and uploads it to Azure Blob Storage.
    Designed to be run concurrently.

    Args:
        year (int): Year of the messages.
        month (int): Month of the messages.
        messages (list): List of message strings for the month.
        master_folder_name (str): Unique folder name for the run.
        container_client (ContainerClient): Initialized Azure ContainerClient.

    Returns:
        tuple: (success_flag (bool), year (int), month (int))
    """
    month_year_str = f"{year}-{month:02d}"
    # Skip processing if a month has no messages (considered success)
    if not messages: return (True, year, month)

    pdf_buffer = None
    try:
        # Generate the raw PDF in memory
        pdf_buffer = create_monthly_pdf(messages, year, month)
        # Define the blob path in Azure
        blob_name = f"{master_folder_name}/messages/{year}/{month:02d}/messages_{year}_{month:02d}.pdf"
        # Upload the PDF buffer
        upload_to_azure(pdf_buffer, container_client, blob_name)
        return (True, year, month) # Indicate success
    except Exception as pdf_or_upload_err:
        # Log errors during PDF creation or upload for this specific month
        print(f"!! Error processing raw log for {month_year_str}: {pdf_or_upload_err}", flush=True)
        return (False, year, month) # Indicate failure
    finally:
        # Ensure the memory buffer is closed
        if pdf_buffer: pdf_buffer.close()


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles the PDF file upload from the user.
    1. Validates the file.
    2. Cleans up any previous session data.
    3. Extracts text and participant names from the PDF.
    4. Cleans the extracted text.
    5. Parses messages into a year/month structure.
    6. Saves the parsed structure to a temporary JSON file.
    7. Stores metadata (participants, folder name, temp file path) in the session.
    8. Starts a PARALLEL process to generate and upload raw monthly message logs to Azure.
    9. Redirects back to the index page with status messages.
    """
    start_time = time.time()
    print("\n--- Received New Upload Request ---", flush=True)

    # --- File Validation ---
    if 'pdf_file' not in request.files:
        flash('No file part in the request.', 'error')
        return redirect(request.url)
    file = request.files['pdf_file']
    if file.filename == '':
        flash('No file selected.', 'error')
        return redirect(request.url)
    if not file.filename.lower().endswith('.pdf'):
        flash('Invalid file type. Please upload a PDF.', 'error')
        return redirect(request.url)

    process_id = None
    temp_file_path = None

    if file:
        # --- Cleanup Previous Session Data ---
        # Remove data associated with any previous upload in this user's session
        old_process_id = session.pop('last_process_id', None)
        if old_process_id:
            print(f"Cleaning up previous session data for ID: {old_process_id}", flush=True)
            old_session_data = session.pop(old_process_id, None)
            # Remove old temporary data file if it exists
            if old_session_data and 'temp_file_path' in old_session_data:
                old_temp_file = old_session_data['temp_file_path']
                if old_temp_file and os.path.exists(old_temp_file):
                    try:
                        os.remove(old_temp_file)
                        print(f"  Removed old temp data file: {old_temp_file}", flush=True)
                    except OSError as e:
                        print(f"  Error removing old temp data file {old_temp_file}: {e}", flush=True)
            # Remove old temporary report directory if it exists
            old_report_temp_dir = os.path.join(TEMP_DATA_DIR, f"{old_process_id}_reports")
            if os.path.isdir(old_report_temp_dir):
                try:
                    shutil.rmtree(old_report_temp_dir)
                    print(f"  Removed old report temp directory: {old_report_temp_dir}", flush=True)
                except OSError as e:
                    print(f"  Error removing old report temp directory {old_report_temp_dir}: {e}", flush=True)
            session.modified = True # Ensure session changes are saved

        # --- Start New Process ---
        # Generate a unique ID for this upload process
        process_id = f"ofw_{uuid.uuid4()}"
        # Create a unique base folder name for Azure uploads for this run
        master_folder_name = f"ofw_extract_{process_id.split('_')[1]}"
        # Define the path for the temporary JSON file storing parsed messages
        temp_file_path = os.path.join(TEMP_DATA_DIR, f"{process_id}_data.json")
        print(f"Starting initial processing for new ID: {process_id}", flush=True)
        print(f"Filename: {file.filename}", flush=True)

        try:
            # --- PDF Processing ---
            print("Reading PDF file into memory...", flush=True)
            pdf_stream_bytes = file.read() # Read entire file into memory
            file.close() # Close the uploaded file handle promptly
            print(f"  File size: {len(pdf_stream_bytes)} bytes.", flush=True)

            # Extract text and participants using the helper function
            full_text, participants = extract_text_and_participants(io.BytesIO(pdf_stream_bytes))
            print(f"Extracted {len(full_text)} characters. Participants detected: '{participants}'", flush=True)

            # Check if any text was extracted
            if not full_text:
                flash('Could not extract any text from the PDF. It might be empty or image-based.', 'error')
                return redirect(url_for('index'))

            # Clean the extracted text
            print("Cleaning extracted text...", flush=True)
            cleaned_full_text = clean_message_text(full_text)
            del full_text # Free memory as soon as possible

            # Parse the cleaned text into messages grouped by year/month
            print("Parsing messages...", flush=True)
            messages_by_year_month = parse_messages(cleaned_full_text)
            del cleaned_full_text # Free memory

            # Check if any messages were successfully parsed
            if not messages_by_year_month:
                flash('No valid messages with parseable dates found in the PDF content.', 'error')
                return redirect(url_for('index'))

            total_months_found = sum(len(months) for months in messages_by_year_month.values())
            print(f"Parsed messages into {len(messages_by_year_month)} years: {sorted(messages_by_year_month.keys())}", flush=True)
            print(f"Total distinct Year/Month combinations found: {total_months_found}", flush=True)

            # --- Save Temporary Data ---
            # Store the parsed message structure locally for the next step (report generation)
            print(f"Saving parsed message structure to temporary file: {temp_file_path}", flush=True)
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                # Convert integer keys (year, month) to strings for JSON compatibility
                serializable_data = {
                    str(year): {str(month): messages for month, messages in months_data.items()}
                    for year, months_data in messages_by_year_month.items()
                }
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            print("Temporary data saved successfully.", flush=True)

            # --- Store Metadata in Session ---
            # Save key information needed for the next step in the user's session
            session['last_process_id'] = process_id
            session[process_id] = {
                'participants': participants,
                'master_folder': master_folder_name, # Folder name in Azure
                'temp_file_path': temp_file_path,   # Path to local data file
                'total_months': total_months_found
            }
            session.modified = True
            print(f"Stored metadata in session for ID: {process_id}", flush=True)

            # --- Parallel Raw Log Upload ---
            # Generate and upload the simple raw message PDFs for each month concurrently
            print("\nConnecting to Azure Blob Storage for raw log upload...", flush=True)
            blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
            container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
            print(f"Connected to Azure container: {AZURE_STORAGE_CONTAINER_NAME}", flush=True)

            # Create list of tasks (one per month) for the thread pool
            tasks = []
            for year, months_data in messages_by_year_month.items():
                for month, messages in months_data.items():
                     tasks.append((year, month, messages))

            upload_count = 0
            upload_errors = 0
            print(f"Starting PARALLEL upload of {len(tasks)} raw monthly message logs using up to {MAX_WORKERS} workers...", flush=True)
            # Use ThreadPoolExecutor for I/O-bound tasks (PDF gen + Azure upload)
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks and store futures
                future_to_month = {
                    executor.submit(generate_and_upload_raw_log, year, month, msgs, master_folder_name, container_client): (year, month)
                    for year, month, msgs in tasks
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_month):
                    year, month = future_to_month[future]
                    try:
                        # Get result: (success_flag, year, month)
                        success, _, _ = future.result()
                        if success: upload_count += 1
                        else: upload_errors += 1
                    except Exception as exc:
                        # Catch errors from the task execution itself
                        print(f"!! Exception in raw log task for {year}-{month:02d}: {exc}", flush=True)
                        upload_errors += 1

                    # Log progress periodically
                    processed_count = upload_count + upload_errors
                    if processed_count % 20 == 0 or processed_count == len(tasks):
                         print(f"  Processed {processed_count}/{len(tasks)} raw log uploads ({upload_errors} errors so far)...", flush=True)

            # --- Finalize Upload Step ---
            end_time = time.time()
            duration = end_time - start_time
            print(f"--- Initial processing & parallel raw log upload complete for {process_id} in {duration:.2f}s ---", flush=True)
            print(f"  Successfully uploaded {upload_count}/{total_months_found} raw monthly logs.")

            # Flash appropriate message to the user based on upload results
            if upload_errors > 0:
                flash(f'Warning: Completed initial PDF processing & raw log uploads ({duration:.2f}s). Successfully uploaded {upload_count} raw logs, but {upload_errors} failed (see server logs). You can proceed to generate reports.', 'warning')
            else:
                flash(f'Successfully processed PDF and uploaded all {upload_count} monthly raw message logs ({duration:.2f}s). You can now generate analysis reports.', 'success')

            # Redirect back to the index page (which will now show the Generate button)
            return redirect(url_for('index'))

        except fitz.fitz.FileDataError as fe:
             # Handle specific error for invalid/corrupted PDFs
             flash(f'Error: The uploaded file does not appear to be a valid PDF or is corrupted. Details: {fe}', 'error')
             # Perform cleanup if PDF processing failed early
             if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path)
             if process_id:
                 session.pop(process_id, None)
                 report_temp_dir = os.path.join(TEMP_DATA_DIR, f"{process_id}_reports")
                 if os.path.isdir(report_temp_dir): shutil.rmtree(report_temp_dir, ignore_errors=True)
             if session.get('last_process_id') == process_id: session.pop('last_process_id', None)
             session.modified = True
             return redirect(url_for('index'))

        except Exception as e:
            # Handle any other unexpected errors during the upload/parse phase
            # Perform general cleanup
            if temp_file_path and os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as rm_err: print(f"Error removing temp file during exception handling: {rm_err}", flush=True)
            if process_id:
                session.pop(process_id, None)
                report_temp_dir = os.path.join(TEMP_DATA_DIR, f"{process_id}_reports")
                if os.path.isdir(report_temp_dir): shutil.rmtree(report_temp_dir, ignore_errors=True)
            if session.get('last_process_id') == process_id: session.pop('last_process_id', None)
            session.modified = True
            flash(f'A critical error occurred during initial PDF processing: {e}', 'error')
            print(f"Detailed Error during upload/parse phase: {e.__class__.__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc() # Log stack trace for debugging
            return redirect(url_for('index'))

    # Fallback redirect if no file was processed (shouldn't normally happen with checks above)
    return redirect(url_for('index'))

# --- Helper function for parallel monthly report processing ---
def process_one_month(year, month, messages, participants, selected_model, temp_report_dir, master_folder_name, container_client):
    """
    Processes a single month: Calls AI for analysis, generates the analysis PDF (locally),
    and uploads the PDF to Azure. Designed to be run concurrently.

    Args:
        year (int): Year of the messages.
        month (int): Month of the messages.
        messages (list): List of message strings for the month.
        participants (str): Names of participants.
        selected_model (str): AI model identifier ('azure' or 'gemini').
        temp_report_dir (str): Base directory for saving the temporary report PDF.
        master_folder_name (str): Unique folder name for the run.
        container_client (ContainerClient): Initialized Azure ContainerClient.

    Returns:
        tuple: (year, month, analysis_data_or_error, status_flag) where status_flag is one of:
               'success': AI analysis and PDF generation/upload succeeded.
               'success_skipped': Month had no messages, skipped processing.
               'analysis_error': AI analysis failed, error report generated/uploaded.
               'pdf_error': AI analysis succeeded, but PDF generation/saving failed.
               'upload_error': AI/PDF succeeded, but Azure upload failed.
               'task_error': An unexpected error occurred within the task itself.
    """
    month_year_str = datetime(year, month, 1).strftime('%B %Y')
    print(f"-- Starting analysis for: {month_year_str}", flush=True)

    analysis_data = None
    local_pdf_path = None
    status_flag = 'analysis_error' # Default status assumes failure until success

    try:
        # --- Handle Empty Months ---
        if not messages:
            # Return immediately for months with no messages
            return (year, month, {"error": "Skipped empty month"}, "success_skipped")

        # --- Prepare AI Input & Truncate if Needed ---
        combined_messages = "\n\n--- MESSAGE SEPARATOR ---\n\n".join(messages)
        original_length = len(combined_messages)
        truncated = False
        if original_length > MAX_INPUT_CHARS:
            print(f"Warning: Truncating input text for {month_year_str} from {original_length} to {MAX_INPUT_CHARS} chars.", flush=True)
            combined_messages = combined_messages[:MAX_INPUT_CHARS]
            # Attempt to truncate at a message separator for cleaner input, otherwise just cut
            last_separator = combined_messages.rfind("\n\n--- MESSAGE SEPARATOR ---\n\n")
            if last_separator != -1:
                combined_messages = combined_messages[:last_separator] + "\n...[TRUNCATED]"
            else:
                combined_messages = combined_messages[:MAX_INPUT_CHARS] + "\n...[TRUNCATED]"
            truncated = True

        # --- Call AI Model ---
        print(f"  Requesting {selected_model.upper()} analysis for {month_year_str}...")
        if selected_model == 'azure':
            analysis_data = call_azure_openai(combined_messages, participants, month_year_str, create_analysis_prompt)
        else: # gemini
            analysis_data = call_gemini_flash(combined_messages, participants, month_year_str, create_analysis_prompt)

        # --- Process AI Response ---
        # analysis_data is now a dict, either with analysis or with {"error": ...}
        if analysis_data.get("error"):
            print(f"!! Failed monthly analysis for {month_year_str}: {analysis_data['error']}", flush=True)
            status_flag = 'analysis_error'
            # analysis_data already contains the error details
        else:
            # AI call was successful
            print(f"  Successfully obtained monthly analysis for {month_year_str}.", flush=True)
            status_flag = 'success' # Tentative success status
            if truncated: print(f"    Note: Analysis for {month_year_str} was based on TRUNCATED message data.", flush=True)

        # --- Generate PDF Report (handles both success and error cases) ---
        # This function will generate a standard report or an error report based on analysis_data
        print(f"  Generating monthly analysis report PDF for {month_year_str}...", flush=True)
        local_pdf_path = create_analysis_report_pdf(
            participants, month_year_str, analysis_data, report_type="Monthly",
            temp_report_dir=temp_report_dir, master_folder_name=master_folder_name,
            selected_model=selected_model, year=year, month=month
        )

        # --- Check PDF Generation & Upload ---
        if not local_pdf_path or not os.path.exists(local_pdf_path):
             # PDF creation or saving failed
             print(f"!! Failed to create or save local PDF for {month_year_str}", flush=True)
             # Update status: if analysis succeeded but PDF failed, it's a 'pdf_error'
             status_flag = 'pdf_error' if status_flag == 'success' else status_flag
             # Return failure status; cannot proceed to upload if PDF doesn't exist
             return (year, month, analysis_data, status_flag)

        # PDF exists locally, proceed with upload
        # Determine correct blob name (append _ERROR if it's an error report)
        analysis_failed_flag = analysis_data.get("error") is not None
        # Check filename in case PDF build failed after successful analysis (gets _BUILD_ERROR suffix)
        is_error_report_pdf = "_ERROR" in os.path.basename(local_pdf_path) or "_BUILD_ERROR" in os.path.basename(local_pdf_path)
        suffix = "_ERROR" if (analysis_failed_flag or is_error_report_pdf) else ""
        # Define Azure blob path
        report_blob_name = f"{master_folder_name}/reports/{selected_model}/{year}/{month:02d}/analysis_report_{year}_{month:02d}{suffix}.pdf"

        print(f"  Uploading monthly report for {month_year_str} to Azure: {report_blob_name}", flush=True)
        # Upload the generated PDF (success or error report)
        upload_to_azure(local_pdf_path, container_client, report_blob_name)

        # Final status check: If analysis was initially successful, but an error report PDF
        # was generated (due to PDF build failure), mark the final status as 'pdf_error'.
        if is_error_report_pdf and status_flag == 'success':
            status_flag = 'pdf_error'

        # If we reached here without upload error, the process for the month is complete.
        print(f"  Finished processing month: {month_year_str} with status: {status_flag}", flush=True)
        return (year, month, analysis_data, status_flag)

    except Exception as processing_err:
         # --- Catch-all for unexpected errors within this month's processing task ---
         print(f"!! Uncaught Error during processing/upload stage for {month_year_str}: {processing_err}", flush=True)
         # Try to determine the stage where the failure occurred
         if local_pdf_path and os.path.exists(local_pdf_path):
             current_status = 'upload_error' # Failed during/after upload attempt
         elif analysis_data and not analysis_data.get("error"):
             current_status = 'pdf_error' # Failed during/after PDF generation
         else:
             current_status = 'analysis_error' # Failed during/before AI call or processing response

         # Ensure analysis_data is a dict and includes error info for reporting
         if not isinstance(analysis_data, dict):
             analysis_data = {"error": f"Processing Task Error: {processing_err}"}
         elif "error" not in analysis_data: # Add error if analysis seemed ok but task failed later
              analysis_data["error"] = f"Processing Task Error: {processing_err}"

         # Return the determined status and error info
         return (year, month, analysis_data, current_status)


# --- Helper function for parallel yearly report processing ---
def process_one_year(year, monthly_data_for_year, participants, selected_model, temp_report_dir, master_folder_name, container_client):
    """
    Processes a single year: Calls AI for yearly synthesis (via helper function),
    generates the yearly report PDF (locally), and uploads it to Azure.
    Designed to be run concurrently.

    Args:
        year (int): The year to process.
        monthly_data_for_year (dict): Dictionary of successful monthly analysis data for this year.
        participants (str): Names of participants.
        selected_model (str): AI model identifier ('azure' or 'gemini').
        temp_report_dir (str): Base directory for saving the temporary report PDF.
        master_folder_name (str): Unique folder name for the run.
        container_client (ContainerClient): Initialized Azure ContainerClient.

    Returns:
        tuple: (year, status_flag) where status_flag is one of:
               'success': Yearly AI analysis and PDF generation/upload succeeded.
               'success_error_report': Yearly AI failed or PDF build failed, but an error report PDF was successfully generated/uploaded.
               'error': Critical failure during yearly processing (e.g., PDF save/upload failed).
    """
    print(f"------ Starting Yearly Processing for: {year} ------", flush=True)
    local_pdf_path = None
    status_flag = 'error' # Default status

    try:
        # --- Generate Yearly Report Data & PDF ---
        # This function handles the AI call for yearly synthesis AND generates the PDF (success or error)
        local_pdf_path = generate_yearly_report_data_and_pdf(
            year, monthly_data_for_year, participants, selected_model,
            master_folder_name, temp_report_dir
        )

        # --- Check PDF Generation Result ---
        if not local_pdf_path or not os.path.exists(local_pdf_path):
             # Failed to create/save the yearly PDF (even an error report)
             print(f"!! Failed to create or save local yearly PDF for {year}", flush=True)
             return (year, 'error') # Return error status for the year

        # --- Determine Blob Name and Upload ---
        # Check if the generated PDF is an error report (based on filename suffix)
        is_error_report = "_ERROR" in os.path.basename(local_pdf_path) or "_BUILD_ERROR" in os.path.basename(local_pdf_path)
        suffix = "_ERROR" if is_error_report else ""
        # Define Azure blob path for the yearly report
        report_blob_name = f"{master_folder_name}/reports/{selected_model}/{year}/yearly_analysis_report_{year}{suffix}.pdf"

        print(f"  Uploading yearly report for {year} to Azure: {report_blob_name}", flush=True)
        # Upload the generated yearly PDF
        upload_to_azure(local_pdf_path, container_client, report_blob_name)

        # Determine final status based on whether a success or error report was uploaded
        status_flag = 'success_error_report' if is_error_report else 'success'
        print(f"------ Finished Yearly Processing for: {year} with status: {status_flag} ------", flush=True)
        return (year, status_flag)

    except Exception as e:
        # --- Catch-all for unexpected errors during this year's processing task ---
        print(f"!! Uncaught Error during yearly processing/upload for {year}: {e}", flush=True)
        import traceback
        traceback.print_exc() # Log stack trace
        return (year, 'error') # Return error status for the year


@app.route('/generate_reports', methods=['POST'])
def generate_reports():
    """
    Handles the report generation request triggered by the user.
    1. Validates the process ID and retrieves session data.
    2. Sets up a temporary directory for report PDFs.
    3. Loads the parsed message data from the temporary JSON file.
    4. Initializes Azure connection.
    5. Starts PARALLEL processing for all MONTHLY reports using `process_one_month`.
    6. Collects results from monthly processing.
    7. Starts PARALLEL processing for all YEARLY reports using `process_one_year`,
       feeding it the successful monthly results.
    8. Collects results from yearly processing.
    9. Generates a final status summary message.
    10. Creates a ZIP archive containing all successfully generated local report PDFs.
    11. Sends the ZIP file to the user as a download.
    12. Performs cleanup (removes temporary files and directories, clears session data).
    """
    start_time = time.time()
    # Get process ID and selected AI model from the form submission
    process_id = request.form.get('process_id')
    selected_model = request.form.get('ai_model', 'gemini') # Default to gemini if not specified
    temp_file_path = None
    report_temp_dir = None
    zip_buffer = None # Initialize zip_buffer in outer scope

    print(f"\n--- Received Report Generation Request (Parallel) ---", flush=True)
    print(f"Process ID: {process_id}, AI Model Selected: {selected_model.upper()}", flush=True)

    # --- Session and Data Validation ---
    if not process_id:
        flash('Process ID missing. Cannot generate reports.', 'error')
        return redirect(url_for('index'))

    # Retrieve data stored in session during the upload step
    stored_session_data = session.get(process_id)
    if not stored_session_data or 'temp_file_path' not in stored_session_data:
        # If session data is missing/incomplete, require user to re-upload
        flash('Session data missing or incomplete for this process. Please upload the PDF again.', 'error')
        session.pop(process_id, None) # Clean up potentially partial session data
        if session.get('last_process_id') == process_id: session.pop('last_process_id', None)
        session.modified = True
        return redirect(url_for('index'))

    # Extract necessary info from session data
    temp_file_path = stored_session_data['temp_file_path']
    participants = stored_session_data.get('participants', 'Unknown')
    master_folder_name = stored_session_data.get('master_folder', f"ofw_extract_{process_id.split('_')[1]}")

    # --- Setup Temporary Directory for Reports ---
    # Create a unique directory for this run to store generated PDFs before zipping
    report_temp_dir = os.path.join(TEMP_DATA_DIR, f"{process_id}_reports")
    try:
        # Clear any existing directory from a previous failed run with the same ID
        if os.path.isdir(report_temp_dir): shutil.rmtree(report_temp_dir)
        os.makedirs(report_temp_dir, exist_ok=True)
        print(f"Created temporary directory for reports: {report_temp_dir}", flush=True)
    except OSError as e:
         # Critical failure if report directory cannot be created
         flash(f'Critical Error: Could not create temporary directory for reports: {e}. Cannot proceed.', 'error')
         # Perform minimal cleanup before redirecting
         session.pop(process_id, None)
         if session.get('last_process_id') == process_id: session.pop('last_process_id', None)
         if temp_file_path and os.path.exists(temp_file_path):
             try: os.remove(temp_file_path)
             except OSError: pass # Ignore error removing temp file here
         session.modified = True
         return redirect(url_for('index'))

    print(f"Starting report generation for Process ID: {process_id}", flush=True)
    print(f"Participants: '{participants}', Master Folder: '{master_folder_name}'", flush=True)

    # --- Load Message Data ---
    # Load the parsed message structure saved during the upload step
    if not os.path.exists(temp_file_path):
        flash(f'Critical Error: Temporary data file not found ({temp_file_path}). Please upload the PDF again.', 'error')
        # Clean up session and report dir if data file is missing
        session.pop(process_id, None)
        if session.get('last_process_id') == process_id: session.pop('last_process_id', None)
        if report_temp_dir and os.path.isdir(report_temp_dir): shutil.rmtree(report_temp_dir, ignore_errors=True)
        session.modified = True
        return redirect(url_for('index'))

    messages_by_year_month = {}
    try:
        print(f"Loading message data from temporary file: {temp_file_path}", flush=True)
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            loaded_data_str_keys = json.load(f)
            # Convert string keys from JSON back to integers for year/month
            messages_by_year_month = {
                int(year_str): {
                    int(month_str): messages
                    for month_str, messages in months_data.items()
                    if month_str.isdigit() and 1 <= int(month_str) <= 12 # Basic validation
                }
                for year_str, months_data in loaded_data_str_keys.items()
                if year_str.isdigit()
            }
            # Remove empty year entries that might result from month filtering
            for year in list(messages_by_year_month.keys()):
                 if not messages_by_year_month[year]:
                     del messages_by_year_month[year]
        # Ensure data was loaded correctly
        if not messages_by_year_month: raise ValueError("Loaded data is empty or contains no valid year/month structure.")
        print("Message data loaded successfully.", flush=True)
    except (json.JSONDecodeError, ValueError, KeyError, Exception) as e:
        # Handle errors loading or parsing the temporary data file
        flash(f'Error loading or parsing temporary data file: {e}. Please upload the PDF again.', 'error')
        # Perform cleanup
        session.pop(process_id, None)
        if session.get('last_process_id') == process_id: session.pop('last_process_id', None)
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except OSError as rm_err: print(f"Error removing corrupt temp file: {rm_err}", flush=True)
        if report_temp_dir and os.path.isdir(report_temp_dir): shutil.rmtree(report_temp_dir, ignore_errors=True)
        session.modified = True
        return redirect(url_for('index'))

    # --- Initialize Counters & Data Structures ---
    monthly_results = {} # Store results as {(year, month): (analysis_data, status_flag)}
    yearly_results = {}  # Store results as {year: status_flag}
    total_months_to_process = sum(len(months) for months in messages_by_year_month.values())
    total_years_to_process = len(messages_by_year_month)
    print(f"\nFound {total_months_to_process} total Month(s) across {total_years_to_process} Year(s) to analyze.", flush=True)

    try:
        # --- Connect to Azure ---
        print("Connecting to Azure Blob Storage for report upload...", flush=True)
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
        print("Connected to Azure.", flush=True)

        # --- Parallel Monthly Processing ---
        print(f"\n======= Starting PARALLEL Monthly Analysis ({total_months_to_process} months) using up to {MAX_WORKERS} workers =======", flush=True)
        # Create list of tasks for monthly processing
        monthly_tasks = []
        for year in sorted(messages_by_year_month.keys()):
            for month in sorted(messages_by_year_month[year].keys()):
                 messages = messages_by_year_month[year][month]
                 monthly_tasks.append((year, month, messages))

        processed_monthly_count = 0
        # Use ThreadPoolExecutor for concurrent monthly processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="MonthWorker") as executor:
            # Submit all monthly tasks
            future_to_month = {
                executor.submit(process_one_month, year, month, msgs, participants, selected_model, report_temp_dir, master_folder_name, container_client): (year, month)
                for year, month, msgs in monthly_tasks
            }

            # Collect results as tasks complete
            for future in concurrent.futures.as_completed(future_to_month):
                year, month = future_to_month[future]
                try:
                    # Result is tuple: (year, month, analysis_data, status)
                    m_year, m_month, m_analysis_data, m_status = future.result()
                    # Store result keyed by (year, month)
                    monthly_results[(m_year, m_month)] = (m_analysis_data, m_status)
                except Exception as exc:
                    # Catch critical errors in the task execution itself
                    print(f"!!! CRITICAL ERROR in monthly task future for {year}-{month:02d}: {exc}", flush=True)
                    # Store error info if task failed catastrophically
                    monthly_results[(year, month)] = ({"error": f"Task execution failed: {exc}"}, "task_error")
                finally:
                    # Log progress
                    processed_monthly_count += 1
                    if processed_monthly_count % 10 == 0 or processed_monthly_count == total_months_to_process:
                        print(f"  Completed {processed_monthly_count}/{total_months_to_process} monthly tasks...", flush=True)

        print("======= Finished PARALLEL Monthly Analysis =======", flush=True)

        # --- Prepare Data for Yearly Analysis ---
        # Filter monthly results to get only successful analyses for yearly synthesis
        yearly_analysis_input_data = defaultdict(lambda: defaultdict(dict))
        successful_months_count = 0
        for (year, month), (analysis_data, status) in monthly_results.items():
             # Include month only if status indicates success AND analysis_data is valid dict without error key
             if status not in ['analysis_error', 'task_error'] and isinstance(analysis_data, dict) and not analysis_data.get("error"):
                 yearly_analysis_input_data[year][month] = analysis_data
                 successful_months_count += 1
             elif status == 'success_skipped': pass # Skipped empty months don't contribute to yearly summary

        years_to_process_yearly = sorted(yearly_analysis_input_data.keys())
        print(f"\nFound {successful_months_count} successful monthly analyses across {len(years_to_process_yearly)} years for yearly aggregation.", flush=True)

        # --- Parallel Yearly Processing ---
        if years_to_process_yearly:
            # Adjust number of workers if fewer years than max workers
            num_yearly_workers = min(MAX_WORKERS, len(years_to_process_yearly))
            print(f"\n======= Starting PARALLEL Yearly Analysis ({len(years_to_process_yearly)} years) using up to {num_yearly_workers} workers =======", flush=True)
            # Create list of tasks for yearly processing
            yearly_tasks = [(year, yearly_analysis_input_data[year]) for year in years_to_process_yearly]
            processed_yearly_count = 0
            # Use ThreadPoolExecutor for concurrent yearly processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_yearly_workers, thread_name_prefix="YearWorker") as executor:
                # Submit all yearly tasks
                future_to_year = {
                    executor.submit(process_one_year, year, monthly_data, participants, selected_model, report_temp_dir, master_folder_name, container_client): year
                    for year, monthly_data in yearly_tasks
                }

                # Collect results as tasks complete
                for future in concurrent.futures.as_completed(future_to_year):
                    year = future_to_year[future]
                    try:
                        # Result is tuple: (year, status_flag)
                        y_year, y_status = future.result()
                        yearly_results[y_year] = y_status # Store status for the year
                    except Exception as exc:
                        # Catch critical errors in the task execution itself
                        print(f"!!! CRITICAL ERROR in yearly task future for {year}: {exc}", flush=True)
                        yearly_results[year] = 'error' # Mark year as failed if task crashed
                    finally:
                         # Log progress
                         processed_yearly_count += 1
                         print(f"  Completed {processed_yearly_count}/{len(years_to_process_yearly)} yearly tasks...", flush=True)

            print("======= Finished PARALLEL Yearly Analysis =======", flush=True)
        else:
            # Skip yearly analysis if no successful monthly data was available
            print("\n======= Skipping Yearly Analysis (No successful monthly data available) =======", flush=True)

        # --- Final Summary and Zipping ---
        end_time = time.time()
        duration = end_time - start_time
        model_name_display = "Google Gemini" if selected_model == 'gemini' else "Azure OpenAI"
        print(f"\n--- Report Generation Process Completed for {process_id} in {duration:.2f}s ---", flush=True)

        # --- Calculate Final Statistics for Summary Message ---
        monthly_processed = total_months_to_process
        # Count outcomes based on status flags returned by process_one_month
        monthly_analysis_ok = sum(1 for _, status in monthly_results.values() if status not in ['analysis_error', 'task_error']) # Analysis attempted and didn't fail immediately
        monthly_analysis_fail = sum(1 for _, status in monthly_results.values() if status == 'analysis_error')
        monthly_pdf_fail = sum(1 for _, status in monthly_results.values() if status == 'pdf_error')
        monthly_upload_fail = sum(1 for _, status in monthly_results.values() if status == 'upload_error')
        monthly_task_fail = sum(1 for _, status in monthly_results.values() if status == 'task_error')
        monthly_fully_successful = sum(1 for _, status in monthly_results.values() if status == 'success' or status == 'success_skipped') # True success or skipped empty

        yearly_attempted = len(years_to_process_yearly) # Years for which yearly analysis was tried
        # Count outcomes based on status flags returned by process_one_year
        yearly_successful = sum(1 for status in yearly_results.values() if status == 'success')
        yearly_error_report_generated = sum(1 for status in yearly_results.values() if status == 'success_error_report') # An error occurred, but error report was generated/uploaded
        yearly_failed = sum(1 for status in yearly_results.values() if status == 'error') # Critical failure for the year
        skipped_years = total_years_to_process - yearly_attempted # Years with no valid monthly data
        total_failures = monthly_analysis_fail + monthly_pdf_fail + monthly_upload_fail + monthly_task_fail + yearly_failed

        # --- Construct Final Status Message for User ---
        final_status_message = f"Report generation finished using {model_name_display} ({duration:.2f}s). "
        final_status_category = 'info' # Default flash message category

        # Add monthly summary details
        final_status_message += f"Monthly: Processed={monthly_processed}, Analysis OK={monthly_analysis_ok}, Full Success={monthly_fully_successful}. "
        monthly_fails_summary = [] # List specific monthly failure types if they occurred
        if monthly_analysis_fail: monthly_fails_summary.append(f"{monthly_analysis_fail} analysis")
        if monthly_pdf_fail: monthly_fails_summary.append(f"{monthly_pdf_fail} PDF gen")
        if monthly_upload_fail: monthly_fails_summary.append(f"{monthly_upload_fail} upload")
        if monthly_task_fail: monthly_fails_summary.append(f"{monthly_task_fail} task")
        if monthly_fails_summary: final_status_message += f"(Monthly Issues: {', '.join(monthly_fails_summary)}). "

        # Add yearly summary details
        final_status_message += f"Yearly: Attempted={yearly_attempted}, Success={yearly_successful}, Error Reports={yearly_error_report_generated}, Failed={yearly_failed}. "
        if skipped_years > 0: final_status_message += f"({skipped_years} year(s) skipped due to no valid monthly data). "

        # Determine overall success/warning/error status category
        if total_failures == 0 and yearly_error_report_generated == 0 and skipped_years == 0:
            # Ideal case: Everything succeeded
            final_status_category = 'success'
            final_status_message = f"Successfully generated and uploaded all reports! ({duration:.2f}s). Monthly: {monthly_fully_successful}/{monthly_processed}. Yearly: {yearly_successful}/{yearly_attempted}. Model: {model_name_display}. Preparing download..."
        elif total_failures > 0 or yearly_error_report_generated > 0:
            # Some issues occurred, but process completed; generated error reports where possible
            final_status_category = 'warning'
            final_status_message += f"Check server logs for details on issues. Error reports were generated where possible. Preparing download..."
        else: # No failures, but maybe skipped years or only info level
             final_status_category = 'info'
             final_status_message += "Preparing download..."

        print(f"Final Status: {final_status_message}")
        # Store final status message in session to be displayed on the index page after redirect
        session['final_status'] = final_status_message
        session['final_status_category'] = final_status_category
        session.modified = True

        # --- Create ZIP Archive of Generated Reports ---
        print(f"--- Creating ZIP archive from reports in {report_temp_dir} ---", flush=True)
        zip_buffer = io.BytesIO() # Create zip file in memory
        zip_filename = f"{master_folder_name}_analysis_reports_{selected_model}.zip"
        files_zipped = 0
        # Use zipfile library to create the archive
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through the temporary report directory structure
            base_path_len = len(report_temp_dir) + len(os.sep) # Length of base path to remove for relative paths in zip
            for root, dirs, files in os.walk(report_temp_dir):
                relative_root = root[base_path_len:] # Path inside the zip relative to the temp report dir root
                for file in files:
                    # Add only PDF files to the zip
                    if file.lower().endswith('.pdf'):
                        local_path = os.path.join(root, file) # Full local path
                        # Determine path inside the zip file (maintaining structure)
                        arcname = os.path.join(relative_root, file)
                        # Handle top-level files correctly (where relative_root is empty)
                        arcname = file if not relative_root else arcname
                        print(f"  Adding to ZIP: {local_path} as {arcname}", flush=True)
                        zipf.write(local_path, arcname=arcname) # Add file to zip
                        files_zipped += 1

        # Check if any files were actually added to the zip
        if files_zipped == 0:
            print("Warning: No report files found in the temporary directory to zip.", flush=True)
            # Update status message if zip is empty
            session['final_status'] += " (Warning: No report files were found/saved locally to include in the ZIP)."
            session.modified = True
            flash(session['final_status'], session.get('final_status_category', 'warning'))
            # Close the empty buffer before redirecting
            if zip_buffer: zip_buffer.close()
            return redirect(url_for('index'))

        print(f"--- ZIP archive created in memory ({files_zipped} files added). Sending file: {zip_filename} ---", flush=True)
        zip_buffer.seek(0) # Rewind buffer to the beginning for sending

        # --- Send ZIP File to User ---
        # Use Flask's send_file to send the in-memory zip buffer as an attachment.
        # Flask/Waitress will handle closing the buffer after sending.
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True, # Prompt user to download
            download_name=zip_filename # Set the filename for the download
        )

    except Exception as e:
        # --- Catch-all for Unexpected Errors during the Main Generation Process ---
        flash(f'An unexpected critical error occurred during the report generation process: {e}', 'error')
        print(f"Detailed CRITICAL Error during report generation process: {e.__class__.__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc() # Log stack trace for debugging
        # Redirect to index page after critical failure
        # Cleanup will be handled by the finally block
        return redirect(url_for('index'))

    finally:
        # --- Cleanup ---
        # This block executes regardless of whether the try block succeeded or failed.
        print("--- Finalizing report generation request (Cleanup) ---", flush=True)

        # Remove temporary data file (parsed messages JSON)
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Removed temporary data file: {temp_file_path}", flush=True)
            except OSError as e:
                print(f"Error removing temporary data file {temp_file_path}: {e}", flush=True)
        else:
             print(f"Temporary data file already removed or not found: {temp_file_path}", flush=True)

        # Remove temporary report directory and its contents
        if report_temp_dir and os.path.isdir(report_temp_dir):
            try:
                shutil.rmtree(report_temp_dir)
                print(f"Removed temporary report directory: {report_temp_dir}", flush=True)
            except OSError as e:
                # Log warning if directory removal fails (e.g., file lock issues)
                print(f"Warning: Error removing temporary report directory {report_temp_dir}: {e}. Manual cleanup might be needed.", flush=True)
        else:
            print(f"Temporary report directory already removed or not found: {report_temp_dir}", flush=True)

        # NOTE: Explicit zip_buffer.close() is intentionally removed here.
        # If send_file was successful, Flask/Waitress manages the buffer lifecycle.
        # If an exception occurred before send_file, the buffer might still be open,
        # but garbage collection will handle it eventually. Closing it here could
        # potentially interfere with an interrupted send_file operation.

        # Clean up session data related to this process
        if process_id in session:
            session.pop(process_id, None)
            print(f"Removed session data for process ID: {process_id}", flush=True)
        # Remove the reference to this process ID as the 'last' one
        if session.get('last_process_id') == process_id:
            session.pop('last_process_id', None)
            print("Removed last_process_id reference.", flush=True)
        session.modified = True # Ensure session changes are saved


# --- Run the App ---
# Entry point for running the Flask application.
if __name__ == '__main__':
    try:
        # Use Waitress, a production-quality WSGI server, if available
        from waitress import serve
        # Set number of worker threads for Waitress (adjust based on server resources & MAX_WORKERS)
        waitress_threads = max(4, MAX_WORKERS * 2) # Ensure at least 4 threads, more if MAX_WORKERS is high
        print(f"--- Starting Waitress server on http://0.0.0.0:5000 with {waitress_threads} threads (Task Workers: {MAX_WORKERS}) ---")
        serve(app, host='0.0.0.0', port=5000, threads=waitress_threads)
    except ImportError:
        # Fallback to Flask's built-in development server if Waitress is not installed
        print("--- Waitress not found. Falling back to Flask development server. ---")
        print("--- For production environments, install waitress: pip install waitress ---")
        # Use threaded=True for basic concurrency with the dev server (suitable for testing)
        # debug=True enables auto-reloading and detailed error pages (DO NOT USE IN PRODUCTION)
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True) 
