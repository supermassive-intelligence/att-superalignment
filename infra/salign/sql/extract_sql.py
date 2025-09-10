import re


def extract_sql(response):
    # If there are multilpe ```sql blocks, grab the content of the last one
    sql_blocks = re.findall(r"```sql(.*?)```", response, re.DOTALL)
    if sql_blocks:
        response = sql_blocks[-1].strip()
    else:
        # If no sql block is found, return the original response
        return response.strip()

    # Grab the final query if there are multiple separated by ;
    response = response.split(";")[0]

    return response.strip()
