from infra.salign.superalignment.get_alternate_queries import get_alternate_queries
from infra.salign.superalignment.get_evidence import get_evidence
from infra.salign.superalignment.get_insights import get_insights

import random

from infra.salign.util.get_config import get_config
from infra.salign.util.prompt_template import PromptTemplate


class EnglishReasoningPrompt:
    def get_name(self):
        return "English Reasoning"

    def forward(self, example, seed):
        prompt = PromptTemplate().user()

        prompt += f"The database name is `{example['database']['db_id']}`.\n"

        #prompt += get_db_profile(example, seed=seed)
        prompt += "\n\n"

        prompt += f"An analyst was asked to write a query to help answer this question: `{example['question']}`\n"

        prompt += example["alignment_prompt"]

        prompt += "\n\n"

        prompt += get_evidence(example, seed=seed)

        alternate_queries_statement = get_alternate_queries(example, seed=seed)

        prompt += alternate_queries_statement

        prompt += get_insights(example, seed=seed)

        prompt += (
            f"Think step by step to answer the question: `{example['question']}`.\n"
        )

        config = get_config()
        maximum_sentences = config["maximum_sentences"]

        start_over_chance = config["start_over_chance"]

        random.seed(seed)

        should_start_over = random.random() < start_over_chance

        if len(alternate_queries_statement) > 0:
            if not should_start_over:
                prompt += f"First, explain the errors in the incorrect queries in at most {maximum_sentences} sentences.\n"
                prompt += " One incorrect query is the best, focus on that one.\n"
                prompt += f"Second, explain how to fix the errors in the best query in at most {maximum_sentences} sentences.\n"
                prompt += " Start by fixing syntax errors.\n"
                prompt += " Next fix queries that return empty results.\n"
                prompt += " Then, fix queries that return incorrect results.\n"
                prompt += "Next, "
            else:
                prompt += "Instead of fixing the errors in the incorrect queries, start over and write a new query that answers the question in a different way.\n"

        prompt += f"Explain in plain english how to write a query that answers the question in at most {maximum_sentences} sentences.\n"
        prompt += f"Finally, write the final correct query in a ```sql\nYOUR_QUERY_HERE\n``` block.\n"
        prompt += PromptTemplate().assistant()

        return prompt
