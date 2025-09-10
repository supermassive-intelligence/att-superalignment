import re

def extract_reasoning(response):
    """
    Extract all the text before the first ```sql code block.
    If no sql block is found, returns the entire response.
    """
    # Match everything before the first ```
    match = re.search(r'(.*?)```sql', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return response.strip()
