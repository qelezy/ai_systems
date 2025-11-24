"""
Парсер для нечётких множеств из файла
"""
from typing import Dict, List


class MembershipFunction:
    """Базовый класс для функций принадлежности"""
    
    def __call__(self, x: float) -> float:
        """Вычисляет степень принадлежности для значения x"""
        raise NotImplementedError


class TriangularMF(MembershipFunction):
    """Треугольная функция принадлежности"""
    
    def __init__(self, a: float, b: float, c: float):
        """
        a, b, c - параметры треугольной функции
        Если d не указан, используется симметричная функция
        """
        self.a = a
        self.b = b
        self.c = c
    
    def __call__(self, x: float) -> float:
        """Вычисляет степень принадлежности"""
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
        """Вычисляет степень принадлежности"""
        if x <= self.a or x >= self.d:
            return 0.0
        elif self.a < x < self.b:
            return (x - self.a) / (self.b - self.a) if self.b != self.a else 0.0
        elif self.b <= x <= self.c:
            return 1.0
        elif self.c < x < self.d:
            return (self.d - x) / (self.d - self.c) if self.d != self.c else 0.0
        return 0.0


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
    
    def __init__(self, name: str):
        self.name = name
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.terms: Dict[str, FuzzySet] = {}
    
    def add_term(self, term_name: str, mf: MembershipFunction):
        """Добавляет терм к переменной"""
        self.terms[term_name] = FuzzySet(term_name, mf)
    
    def update_range(self, values: List[float]):
        """Обновляет диапазон переменной по параметрам функции принадлежности"""
        if not values:
            return
        
        min_val = min(values)
        max_val = max(values)
        
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


class FuzzySetsParser:
    """Парсер для файла с определениями нечётких множеств"""
    
    def __init__(self):
        self.variables: Dict[str, FuzzyVariable] = {}
    
    def parse_file(self, file_path: str):
        """Парсит файл с определениями нечётких множеств"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                self._parse_line(line)
    
    def _parse_line(self, line: str):
        """Парсит одну строку определения нечёткого множества"""
        # Формат: переменная:терм:тип_функции:параметры
        parts = line.split(':')
        if len(parts) < 4:
            return
        
        var_name = parts[0].strip()
        term_name = parts[1].strip()
        mf_type = parts[2].strip().lower()
        params_str = parts[3].strip()
        
        # Парсим параметры
        params = [float(p.strip()) for p in params_str.split(',')]
        
        # Создаём функцию принадлежности
        mf = self._create_membership_function(mf_type, params)
        
        # Добавляем переменную, если её ещё нет
        if var_name not in self.variables:
            self.variables[var_name] = FuzzyVariable(var_name)
        
        variable = self.variables[var_name]
        variable.add_term(term_name, mf)
        variable.update_range(params)
    
    def _create_membership_function(self, mf_type: str, params: List[float]) -> MembershipFunction:
        """Создаёт функцию принадлежности по типу и параметрам"""
        if mf_type == 'triangular':
            if len(params) == 3:
                return TriangularMF(params[0], params[1], params[2])
        elif mf_type == 'trapezoidal':
            if len(params) == 4:
                return TrapezoidalMF(params[0], params[1], params[2], params[3])
        
        raise ValueError(f"Неизвестный тип функции принадлежности: {mf_type} или неверное количество параметров. Поддерживаются только triangular и trapezoidal.")
    
    def get_variable(self, name: str) -> FuzzyVariable:
        """Возвращает переменную по имени"""
        return self.variables.get(name)
    
    def get_all_variables(self) -> Dict[str, FuzzyVariable]:
        """Возвращает все переменные"""
        return self.variables.copy()

