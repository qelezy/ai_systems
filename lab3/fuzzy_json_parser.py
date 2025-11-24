import json
from typing import Dict, List, Tuple, Union
from fuzzy_inference_engine import FuzzyRule


# ==============================
# ФУНКЦИИ ПРИНАДЛЕЖНОСТИ
# ==============================

class MembershipFunction:
    def __call__(self, x: float) -> float:
        raise NotImplementedError


class TriangularMF(MembershipFunction):
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x: float) -> float:
        if x <= self.a or x >= self.c:
            return 0.0
        elif self.a < x < self.b:
            return (x - self.a) / (self.b - self.a) if self.b != self.a else 0.0
        elif self.b <= x <= self.c:
            return (self.c - x) / (self.c - self.b) if self.c != self.b else 0.0
        return 0.0


class TrapezoidalMF(MembershipFunction):
    def __init__(self, a: float, b: float, c: float, d: float):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, x: float) -> float:
        if x <= self.a or x >= self.d:
            return 0.0
        elif self.a < x < self.b:
            return (x - self.a) / (self.b - self.a) if self.b != self.a else 0.0
        elif self.b <= x <= self.c:
            return 1.0
        elif self.c < x < self.d:
            return (self.d - x) / (self.d - self.c) if self.d != self.c else 0.0
        return 0.0


# ==============================
# НЕЧЁТКИЕ МНОЖЕСТВА И ПЕРЕМЕННЫЕ
# ==============================

class FuzzySet:
    def __init__(self, name: str, mf: MembershipFunction):
        self.name = name
        self.mf = mf

    def membership(self, x: float) -> float:
        return self.mf(x)


class FuzzyVariable:
    def __init__(self, name: str, min_val: float, max_val: float):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.terms: Dict[str, FuzzySet] = {}

    def add_term(self, term_name: str, mf: MembershipFunction):
        self.terms[term_name] = FuzzySet(term_name, mf)

    def get_membership(self, term_name: str, x: float) -> float:
        return self.terms[term_name].membership(x)

    def fuzzify(self, x: float) -> Dict[str, float]:
        return {term: fs.membership(x) for term, fs in self.terms.items()}


# ==============================
# ГЛАВНЫЙ JSON-ПАРСЕР
# ==============================

class JSONFuzzyModelParser:

    def __init__(self):
        self.variables: Dict[str, FuzzyVariable] = {}
        self.rules: List[FuzzyRule] = []
        self.input_variables: List[str] = []
        self.output_variable: str | None = None

    # ---- PUBLIC ----

    def parse_file(self, file_path: str) -> Tuple[Dict[str, FuzzyVariable], List[FuzzyRule], List[str], str]:
        """Считывает модель из JSON и возвращает переменные, правила и имена входов/выхода"""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        model = data["model"]

        # Сбрасываем состояние перед парсингом
        self.variables = {}
        self.rules = []
        self.input_variables = []
        self.output_variable = None

        # Парсим входные переменные (может быть словарь или список словарей)
        inputs_block: Union[dict, List[dict]] = model["input"]
        if isinstance(inputs_block, dict):
            inputs_block = [inputs_block]

        for input_def in inputs_block:
            variable = self._parse_variable(input_def)
            self.input_variables.append(variable.name)

        # Выходная переменная
        output_variable = self._parse_variable(model["output"])
        self.output_variable = output_variable.name

        # Парсим правила
        self._parse_rules(model["rules"])

        return self.variables, self.rules, list(self.input_variables), self.output_variable

    # ---- PRIVATE ----

    def _parse_variable(self, var_json: dict) -> FuzzyVariable:
        name = var_json["name"]
        min_v, max_v = var_json["range"]

        variable = FuzzyVariable(name, min_v, max_v)

        for term_name, term_data in var_json["terms"].items():
            mf_type = term_data["type"]
            params = term_data["params"]
            mf = self._create_mf(mf_type, params)
            variable.add_term(term_name, mf)

        self.variables[name] = variable
        return variable

    def _create_mf(self, mf_type: str, params: List[float]) -> MembershipFunction:
        mf_type = mf_type.lower()

        if mf_type == "tri" and len(params) == 3:
            return TriangularMF(*params)

        if mf_type == "trap" and len(params) == 4:
            return TrapezoidalMF(*params)

        raise ValueError(f"Неверный тип функции принадлежности '{mf_type}' или параметры: {params}")

    def _parse_rules(self, rules_json: List[dict]):
        for rule in rules_json:
            if_block: dict = rule["if"]
            then_block: dict = rule["then"]

            # IF может содержать одну или несколько переменных
            conditions = {var: term for var, term in if_block.items()}

            # THEN предполагаем одну переменную
            res_var, res_term = list(then_block.items())[0]

            self.rules.append(FuzzyRule(conditions, res_var, res_term))
