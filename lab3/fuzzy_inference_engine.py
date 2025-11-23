"""
Движок нечёткого логического вывода

Реализует:
- Типы импликаций: Мамдани, Ларсен
- Композиционные механизмы (max-min, max-product)
- Операторы агрегации (max, sum)
- Механизмы вывода:
  * Композиционный (max-min композиция)
  * Композиционный (max-product композиция)
  * Использование уровней истинности предпосылок правил
- Дефазификацию методом центра тяжести
"""
from typing import Dict, List
import numpy as np
from enum import Enum
from fuzzy_sets_parser import FuzzyVariable


class ImplicationType(Enum):
    """Типы импликаций"""
    MAMDANI = "mamdani"  # min
    LARSEN = "larsen"    # product


class AggregationType(Enum):
    """Типы операторов агрегации"""
    MAX = "max"          # максимум
    SUM = "sum"          # сумма (с ограничением до 1)
    PROBOR = "probor"    # вероятностное ИЛИ: a + b - a * b


class CompositionType(Enum):
    """Типы композиционных механизмов"""
    MAX_MIN = "max_min"      # max-min композиция
    MAX_PROD = "max_prod"    # max-product композиция


class InferenceMechanism(Enum):
    """Механизмы логического вывода"""
    COMPOSITION_MAX_MIN = "composition_max_min"
    COMPOSITION_MAX_PROD = "composition_max_prod"
    TRUTH_LEVEL = "truth_level"


class FuzzyRule:
    """Нечёткое правило"""
    
    def __init__(self, conditions: Dict[str, str], result_var: str, result_term: str):
        self.conditions = conditions
        self.result_var = result_var
        self.result_term = result_term
    
    def __str__(self):
        cond_str = " И ".join([f"{var}={term}" for var, term in self.conditions.items()])
        return f"ЕСЛИ {cond_str} ТО {self.result_var}={self.result_term}"


class FuzzyInferenceEngine:
    """Движок нечёткого логического вывода"""
    
    def __init__(self, rules: List[FuzzyRule], variables: Dict[str, FuzzyVariable]):
        self.rules = rules
        self.variables = variables
    
    def _implication(self, a: float, b: float, impl_type: ImplicationType) -> float:
        """Применяет импликацию"""
        if impl_type == ImplicationType.MAMDANI:
            return min(a, b)
        elif impl_type == ImplicationType.LARSEN:
            return a * b
    
    def _aggregate(self, values: List[float], agg_type: AggregationType) -> float:
        """Применяет оператор агрегации"""
        if not values:
            return 0.0
        
        if agg_type == AggregationType.MAX:
            return max(values)
        elif agg_type == AggregationType.SUM:
            return min(1.0, sum(values))
        elif agg_type == AggregationType.PROBOR:
            result = values[0]
            for v in values[1:]:
                result = result + v - result * v
            return min(1.0, result)
    
    def _evaluate_rule_conditions(self, rule: FuzzyRule, inputs: Dict[str, float]) -> float:
        """Вычисляет уровень истинности предпосылок правила"""
        truth_levels = []
        
        for var_name, term_name in rule.conditions.items():
            if var_name not in inputs:
                return 0.0
            
            var = self.variables.get(var_name)
            if not var:
                return 0.0
            
            truth = var.get_membership(term_name, inputs[var_name])
            truth_levels.append(truth)
        
        # Используем минимум для И (можно расширить для ИЛИ)
        return min(truth_levels) if truth_levels else 0.0
    
    def inference_composition(self, inputs: Dict[str, float], 
                             output_var: str,
                             impl_type: ImplicationType = ImplicationType.MAMDANI,
                             comp_type: CompositionType = CompositionType.MAX_MIN,
                             agg_type: AggregationType = AggregationType.MAX,
                             resolution: int = 1000) -> np.ndarray:
        """
        Композиционный механизм вывода
        
        Args:
            inputs: входные значения {имя_переменной: значение}
            output_var: имя выходной переменной
            impl_type: тип импликации
            comp_type: тип композиции
            agg_type: тип агрегации
            resolution: разрешение для выходной функции
        
        Returns:
            массив значений функции принадлежности выходной переменной
        """
        output_variable = self.variables.get(output_var)
        if not output_variable:
            raise ValueError(f"Переменная {output_var} не найдена")
        
        # Создаём массив значений для выходной переменной
        x_range = np.linspace(output_variable.min_val, output_variable.max_val, resolution)
        output_mf = np.zeros(resolution)
        
        # Обрабатываем каждое правило
        for rule in self.rules:
            if rule.result_var != output_var:
                continue
            
            # Вычисляем уровень истинности предпосылок
            truth_level = self._evaluate_rule_conditions(rule, inputs)
            
            if truth_level == 0.0:
                continue
            
            # Получаем функцию принадлежности выходного терма
            result_term_mf = output_variable.terms.get(rule.result_term)
            if not result_term_mf:
                continue
            
            # Вычисляем функцию принадлежности для выходного терма
            term_membership = np.array([result_term_mf.membership(x) for x in x_range])
            
            # Применяем импликацию
            if comp_type == CompositionType.MAX_MIN:
                rule_output = np.array([self._implication(truth_level, m, impl_type) 
                                       for m in term_membership])
            elif comp_type == CompositionType.MAX_PROD:
                rule_output = truth_level * term_membership
            
            # Агрегируем с предыдущими правилами
            if agg_type == AggregationType.MAX:
                output_mf = np.maximum(output_mf, rule_output)
            elif agg_type == AggregationType.SUM:
                output_mf = np.minimum(1.0, output_mf + rule_output)
            elif agg_type == AggregationType.PROBOR:
                output_mf = output_mf + rule_output - output_mf * rule_output
                output_mf = np.minimum(1.0, output_mf)
        
        return output_mf, x_range
    
    def inference_truth_level(self, inputs: Dict[str, float],
                              output_var: str,
                              impl_type: ImplicationType = ImplicationType.MAMDANI,
                              agg_type: AggregationType = AggregationType.MAX,
                              resolution: int = 1000) -> np.ndarray:
        """
        Механизм вывода с использованием уровней истинности предпосылок правил
        
        Args:
            inputs: входные значения
            output_var: имя выходной переменной
            impl_type: тип импликации
            agg_type: тип агрегации
            resolution: разрешение
        
        Returns:
            массив значений функции принадлежности
        """
        output_variable = self.variables.get(output_var)
        if not output_variable:
            raise ValueError(f"Переменная {output_var} не найдена")
        
        x_range = np.linspace(output_variable.min_val, output_variable.max_val, resolution)
        output_mf = np.zeros(resolution)
        
        # Обрабатываем каждое правило
        for rule in self.rules:
            if rule.result_var != output_var:
                continue
            
            # Вычисляем уровень истинности предпосылок
            truth_level = self._evaluate_rule_conditions(rule, inputs)
            
            if truth_level == 0.0:
                continue
            
            # Получаем функцию принадлежности выходного терма
            result_term_mf = output_variable.terms.get(rule.result_term)
            if not result_term_mf:
                continue
            
            # Вычисляем функцию принадлежности для выходного терма
            term_membership = np.array([result_term_mf.membership(x) for x in x_range])
            
            # Применяем импликацию (обрезаем функцию принадлежности)
            rule_output = np.array([self._implication(truth_level, m, impl_type) 
                                   for m in term_membership])
            
            # Агрегируем
            if agg_type == AggregationType.MAX:
                output_mf = np.maximum(output_mf, rule_output)
            elif agg_type == AggregationType.SUM:
                output_mf = np.minimum(1.0, output_mf + rule_output)
            elif agg_type == AggregationType.PROBOR:
                output_mf = output_mf + rule_output - output_mf * rule_output
                output_mf = np.minimum(1.0, output_mf)
        
        return output_mf, x_range
    
    def defuzzify_centroid(self, membership: np.ndarray, x_range: np.ndarray) -> float:
        """
        Дефазификация методом центра тяжести
        
        Args:
            membership: массив значений функции принадлежности
            x_range: массив значений по оси X
        
        Returns:
            чёткое значение
        """
        # Избегаем деления на ноль
        if np.sum(membership) == 0:
            return np.mean(x_range)
        
        return np.sum(x_range * membership) / np.sum(membership)
    
    def defuzzify_bisector(self, membership: np.ndarray, x_range: np.ndarray) -> float:
        """Дефазификация методом биссектрисы"""
        total_area = np.sum(membership)
        if total_area == 0:
            return np.mean(x_range)
        
        cumulative = np.cumsum(membership)
        half_area = total_area / 2.0
        
        idx = np.searchsorted(cumulative, half_area)
        if idx >= len(x_range):
            idx = len(x_range) - 1
        
        return x_range[idx]
    
    def defuzzify_mom(self, membership: np.ndarray, x_range: np.ndarray) -> float:
        """Дефазификация методом среднего максимума (Mean of Maximum)"""
        max_val = np.max(membership)
        if max_val == 0:
            return np.mean(x_range)
        
        max_indices = np.where(membership == max_val)[0]
        return np.mean(x_range[max_indices])

