import re

def clean_text(text):
    # 1. Remove all kinds of dates (formats: 04/07/2025, July 4, 2025, 2025-07-04)
    date_patterns = [
        r'\b\d{2}/\d{2}/\d{4}\b',            # DD/MM/YYYY
        r'\b\d{4}-\d{2}-\d{2}\b',            # YYYY-MM-DD
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b',  # Month D, YYYY
    ]
    for pattern in date_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 2. Remove email addresses
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)

    # 3. Replace multiple spaces or newlines with a single space
    text = re.sub(r'\s+', ' ', text)

    # 4. Remove all emojis or non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')

    # 5. Remove excessive punctuation (e.g., "!!!", "...", "??")
    text = re.sub(r'([!?.])\1+', r'\1', text)  # Reduce repeated punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)    # Remove other non-word punctuation

    return text.strip()

# Example input
input_text = """
Hey! Contact me at person@example.com.  
The event was on 04/07/2025, and another one on July 4, 2025. 
Visit for more info... 
We also met on 2025-07-04. 
Call me!!!    Or not... 😅😅   
"""

# Clean the text
cleaned_text = clean_text(input_text)
print(cleaned_text)
