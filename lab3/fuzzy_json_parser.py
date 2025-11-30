"""
Единый парсер для нечётких систем из JSON файла
Объединяет функциональность парсеров множеств и правил
"""
import json
from typing import Dict, List, Tuple, Union


# ==============================
# ФУНКЦИИ ПРИНАДЛЕЖНОСТИ
# ==============================

class MembershipFunction:
    """Базовый класс для функций принадлежности"""
    
    def __call__(self, x: float) -> float:
        """Вычисляет степень принадлежности для значения x"""
        raise NotImplementedError


class TriangularMF(MembershipFunction):
    """Треугольная функция принадлежности"""
    
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
    """Трапециевидная функция принадлежности"""
    
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
    """Нечёткое множество"""
    
    def __init__(self, name: str, mf: MembershipFunction):
        self.name = name
        self.mf = mf

    def membership(self, x: float) -> float:
        """Возвращает степень принадлежности значения x"""
        return self.mf(x)


class FuzzyVariable:
    """Нечёткая переменная с несколькими термами"""
    
    def __init__(self, name: str, min_val: float = None, max_val: float = None):
        self.name = name
        # Если min_val/max_val не указаны, вычисляем из термов
        if min_val is not None and max_val is not None:
            self.min_val = min_val
            self.max_val = max_val
        else:
            self.min_val = float('inf')
            self.max_val = float('-inf')
        self.terms: Dict[str, FuzzySet] = {}

    def add_term(self, term_name: str, mf: MembershipFunction):
        """Добавляет терм к переменной"""
        self.terms[term_name] = FuzzySet(term_name, mf)
        # Обновляем диапазон на основе параметров функции принадлежности
        self._update_range_from_mf(mf)

    def _update_range_from_mf(self, mf: MembershipFunction):
        """Обновляет диапазон переменной по параметрам функции принадлежности"""
        params = []
        if isinstance(mf, TriangularMF):
            params = [mf.a, mf.b, mf.c]
        elif isinstance(mf, TrapezoidalMF):
            params = [mf.a, mf.b, mf.c, mf.d]
        
        if params:
            min_val = min(params)
            max_val = max(params)
            
            if self.min_val == float('inf') or min_val < self.min_val:
                self.min_val = min_val
            if self.max_val == float('-inf') or max_val > self.max_val:
                self.max_val = max_val

    def get_membership(self, term_name: str, x: float) -> float:
        """Возвращает степень принадлежности для терма"""
        if term_name not in self.terms:
            return 0.0
        return self.terms[term_name].membership(x)

    def fuzzify(self, x: float) -> Dict[str, float]:
        """Фаззификация: возвращает степени принадлежности для всех термов"""
        return {term: self.get_membership(term, x) for term in self.terms}


# ==============================
# НЕЧЁТКИЕ ПРАВИЛА
# ==============================

class FuzzyRule:
    """Нечёткое правило"""
    
    def __init__(self, conditions: Dict[str, str], result_var: str, result_term: str):
        self.conditions = conditions
        self.result_var = result_var
        self.result_term = result_term
    
    def __str__(self):
        cond_str = " И ".join([f"{var}={term}" for var, term in self.conditions.items()])
        return f"ЕСЛИ {cond_str} ТО {self.result_var}={self.result_term}"


# ==============================
# ГЛАВНЫЙ JSON-ПАРСЕР
# ==============================

class JSONFuzzyModelParser:
    """Единый парсер для нечётких систем из JSON файла"""

    def __init__(self):
        self.variables: Dict[str, FuzzyVariable] = {}
        self.rules: List[FuzzyRule] = []
        self.input_variables: List[str] = []
        self.output_variable: str | None = None

    def parse_file(self, file_path: str) -> Tuple[Dict[str, FuzzyVariable], List[FuzzyRule], List[str], str]:
        """
        Считывает модель из JSON и возвращает переменные, правила и имена входов/выхода
        
        Returns:
            кортеж (словарь переменных, список правил, список имен входных переменных, имя выходной переменной)
        """
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

    def _parse_variable(self, var_json: dict) -> FuzzyVariable:
        """Парсит переменную из JSON"""
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
        """Создаёт функцию принадлежности по типу и параметрам"""
        mf_type = mf_type.lower()

        if mf_type == "tri" and len(params) == 3:
            return TriangularMF(*params)

        if mf_type == "trap" and len(params) == 4:
            return TrapezoidalMF(*params)

        raise ValueError(f"Неверный тип функции принадлежности '{mf_type}' или параметры: {params}")

    def _parse_rules(self, rules_json: List[dict]):
        """Парсит правила из JSON"""
        for rule in rules_json:
            if_block: dict = rule["if"]
            then_block: dict = rule["then"]

            # IF может содержать одну или несколько переменных
            conditions = {var: term for var, term in if_block.items()}

            # THEN предполагаем одну переменную
            res_var, res_term = list(then_block.items())[0]

            self.rules.append(FuzzyRule(conditions, res_var, res_term))
