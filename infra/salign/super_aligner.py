from infra.salign.engine.super_aligner_engine import SuperAlignerEngine


class SuperAligner:
    def __init__(self, llm=None):
        self.engine = SuperAlignerEngine(llm=llm)

    def connect(self, database):
        self.engine.connect(database)

    def load_query_logs(self, query_logs):
        self.engine.load_query_logs(query_logs)

    def load_problems(self, problems):
        self.engine.load_problems(problems)

    def learn_reasoners(self, reasoners):
        self.engine.learn_reasoners(reasoners)

    def align_prompt(self, prompt):
        self.engine.align_prompt(prompt)

    def align(self):
        return self.engine.align()

    def solve(self):
        return self.engine.solve()
