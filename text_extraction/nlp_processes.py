from datetime import datetime
import re

# Helper functions
def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def remove_hidden_characters(text):
    """Remove hidden characters such as non-breaking spaces and zero-width spaces."""
    return ''.join(char for char in text if not char.isspace() or char == ' ').strip()

def log_character_codes(text):
    """Log character codes for analysis."""
    return [ord(char) for char in text]

# Main matching function
def fuzzy_match_with_hidden_char_removal(extracted_texts, input_data, max_distance=2):
    """Enhanced validation with removal of hidden characters, OCR corrections, and detailed date parsing."""
    extracted_info = {'name': False, 'date': False, 'brand': False}

    def normalize_text(text):
        """Normalize text for comparison by removing punctuation and extra spaces."""
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text)).strip().lower()

    def is_match_by_levenshtein(text1, text2):
        """Check if two strings match within a maximum Levenshtein distance."""
        return levenshtein_distance(text1, text2) <= max_distance

    def token_match(text, input_value):
        """Match input text against tokens of the extracted text."""
        tokens = text.split()
        for token in tokens:
            if is_match_by_levenshtein(token, input_value):
                return True
        return False

    def parse_and_match_dates(extracted_text, input_date):
        """Validate dates directly in the format YYYYMMDD with hidden character removal."""
        extracted_text = re.sub(r'^\b(date|brand|name)\b\s*', '', extracted_text, flags=re.IGNORECASE)

        # Strip and normalize the extracted text to remove hidden characters and spaces
        extracted_text = remove_hidden_characters(extracted_text)

        # Directly search for an 8-digit numeric date (YYYYMMDD)
        date_pattern = r'\b\d{8}\b'  # Matches an 8-digit number, e.g., 10102024

        date_match = re.search(date_pattern, extracted_text)
        if date_match:
            # Extract the date and prepare it for comparison
            extracted_date = date_match.group()

            # Format the input date similarly for comparison
            formatted_input_date = remove_hidden_characters(input_date)

            # Direct comparison of formatted dates
            if extracted_date == formatted_input_date:
                return True

        return False

    normalized_input_data = {key: normalize_text(value) for key, value in input_data.items()}

    for text in extracted_texts:
        text_normalized = normalize_text(text)

        # Name matching: Use token matching for multi-word comparisons
        name_tokens = normalized_input_data['name'].split()
        if all(any(is_match_by_levenshtein(token, name_part) for token in text_normalized.split()) for name_part in name_tokens):
            extracted_info['name'] = True

        # Date matching: Flexible parsing with hidden character removal
        if parse_and_match_dates(text_normalized, normalized_input_data['date']):
            extracted_info['date'] = True

        # Brand matching with token-based and Levenshtein distance
        if token_match(text_normalized, normalized_input_data['brand']):
            extracted_info['brand'] = True

    return extracted_info

# Example usage
# extracted_texts_example = ['Date 10-10-2024', 'Brand Netflix .', 'Name dssd Abdukkadir Pasda']
# input_data_example = {'name': 'Abdulkadir Pasa', 'date': '10-10-2024', 'brand': 'Netflix'}
# validated_info = fuzzy_match_with_hidden_char_removal(extracted_texts_example, input_data_example)
# print(validated_info)
