"""
Движок нечёткого логического вывода

Реализует:
- Типы импликаций: Мамдани, Ларсен
  * Мамдани автоматически использует max-min композицию
  * Ларсен автоматически использует max-product композицию
- Операторы агрегации (max, sum)
- Механизмы вывода:
  * Композиционный (тип композиции определяется по типу импликации)
  * Использование уровней истинности предпосылок правил
- Дефазификацию методом центра тяжести
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum
from fuzzy_json_parser import FuzzyVariable, FuzzyRule


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
    MAX_MIN = "max_min"      # max-min композиция (использует min для импликации)
    MAX_PROD = "max_prod"    # max-product композиция (использует product для импликации)

class InferenceMechanism(Enum):
    """Механизмы логического вывода"""
    COMPOSITION_MAX_MIN = "composition_max_min"
    COMPOSITION_MAX_PROD = "composition_max_prod"
    TRUTH_LEVEL = "truth_level"


# FuzzyRule теперь импортируется из fuzzy_json_parser


class FuzzyInferenceEngine:
    """Движок нечёткого логического вывода"""
    
    def __init__(
        self,
        rules: List[FuzzyRule],
        variables: Dict[str, FuzzyVariable],
        condition_resolution: int = 201
    ):
        self.rules = rules
        self.variables = variables
        self.condition_resolution = max(51, condition_resolution)
    
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
    
    def _build_input_membership(self, var: FuzzyVariable, value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Формирует нечёткое множество A'j(x) для входного значения"""
        x_range = np.linspace(var.min_val, var.max_val, self.condition_resolution)
        
        # Для crisp значения строим узкую треугольную ступеньку (фаззифицированный синглтон)
        span = (var.max_val - var.min_val) or 1.0
        half_width = span / max(self.condition_resolution // 2, 1)
        
        membership = np.maximum(1.0 - np.abs(x_range - value) / half_width, 0.0)
        return x_range, membership
    
    def _evaluate_rule_conditions(self, rule: FuzzyRule, inputs: Dict[str, float]) -> float:
        """Вычисляет уровень истинности предпосылок правила"""
        truth_levels = []
        
        for var_name, term_name in rule.conditions.items():
            if var_name not in inputs:
                return 0.0
            
            var = self.variables.get(var_name)
            if not var:
                return 0.0
            
            result_term = var.terms.get(term_name)
            if not result_term:
                truth_levels.append(0.0)
                continue
            
            x_range, input_mf = self._build_input_membership(var, inputs[var_name])
            term_membership = np.array([result_term.membership(x) for x in x_range])
            intersection = np.minimum(input_mf, term_membership)
            truth = float(np.max(intersection))
            truth_levels.append(truth)
        
        # Используем минимум для И (можно расширить для ИЛИ)
        return min(truth_levels) if truth_levels else 0.0
    
    def inference_composition(self, inputs: Dict[str, float], 
                             output_var: str,
                             comp_type: CompositionType,
                             impl_type: ImplicationType = ImplicationType.MAMDANI,
                             agg_type: AggregationType = AggregationType.MAX,
                             resolution: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Композиционный механизм вывода
        
        Правильный алгоритм (согласно теории нечетких систем):
        
        ШАГ 1: Вычисление нечетких соответствий для каждого правила R_i = A_i → B_i
               (определяется типом импликации impl_type)
               - Мамдани: μ_R_i(x,y) = min(α_i, μ_B_i(y))
               - Ларсен: μ_R_i(x,y) = α_i * μ_B_i(y)
               где α_i - уровень истинности предпосылок
        
        ШАГ 2: Вычисление выходов для каждого правила B'_i = A' ○ R_i 
               (определяется типом композиции comp_type)
               - MAX-MIN: B'_i(y) = max_x[min(α_i(x), μ_R_i(x,y))]
                         Упрощённо: B'_i(y) = min(α_i, μ_B_i(y))
               - MAX-PROD: B'_i(y) = max_x[α_i(x) * μ_R_i(x,y)]
                          Упрощённо: B'_i(y) = α_i * μ_B_i(y)
        
        ШАГ 3: Агрегация индивидуальных выходов B' = Agg(B'_1, ..., B'_n)
               (определяется типом агрегации agg_type)
        
        Args:
            inputs: входные значения {имя_переменной: значение}
            output_var: имя выходной переменной
            comp_type: правило композиции (MAX_MIN или MAX_PROD)
            impl_type: тип импликации (Мамдани или Ларсен)
            agg_type: тип агрегации (MAX, SUM, PROBOR)
            resolution: разрешение для выходной функции
        
        Returns:
            кортеж (массив значений функции принадлежности, массив значений x)
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

            # ШАГ 1: Вычисление нечётких соответствий R_i = A_i → B_i
            # Вычисляем уровень истинности предпосылок (α_i)
            truth_level = self._evaluate_rule_conditions(rule, inputs)
            if truth_level == 0.0:
                continue

            # Получаем функцию принадлежности выходного терма μ_B_i(y)
            result_term_mf = output_variable.terms.get(rule.result_term)
            if not result_term_mf:
                continue

            term_membership = np.array([result_term_mf.membership(x) for x in x_range])

            # ШАГ 1 (продолжение): Вычисляем нечёткое соответствие R_i согласно типу импликации
            # R_i(α_i, y) - это функция, которая связывает уровень истинности с выходным термом
            if impl_type == ImplicationType.MAMDANI:
                # Мамдани: R_i(y) = min(α_i, μ_B_i(y))
                # Это нечёткое соответствие, которое будет использоваться в композиции
                fuzzy_relation = np.minimum(truth_level, term_membership)
            elif impl_type == ImplicationType.LARSEN:
                # Ларсен: R_i(y) = α_i * μ_B_i(y)
                # Это нечёткое соответствие, которое будет использоваться в композиции
                fuzzy_relation = truth_level * term_membership
            else:
                fuzzy_relation = np.minimum(truth_level, term_membership)

            # ШАГ 2: Вычисление выходов для каждого правила B'_i = A' ○ R_i
            # Композиция A' с R_i согласно правилу композиции (comp_type)
            # Поскольку A' - это синглтон α_i, композиция упрощается
            if comp_type == CompositionType.MAX_MIN:
                # MAX-MIN композиция: B'_i(y) = max_x[min(α_i, μ_R_i(x,y))]
                # Упрощённо (для синглтона): B'_i(y) = min(α_i, μ_B_i(y))
                # Это уже содержится в fuzzy_relation для Мамдани
                if impl_type == ImplicationType.MAMDANI:
                    rule_output = fuzzy_relation  # уже min(α_i, μ_B_i(y))
                else:
                    # Если impl_type = Ларсен, но comp_type = MAX_MIN, используем min
                    rule_output = np.minimum(truth_level, term_membership)
            elif comp_type == CompositionType.MAX_PROD:
                # MAX-PROD композиция: B'_i(y) = max_x[α_i * μ_R_i(x,y)]
                # Упрощённо (для синглтона): B'_i(y) = α_i * μ_B_i(y)
                # Это уже содержится в fuzzy_relation для Ларсена
                if impl_type == ImplicationType.LARSEN:
                    rule_output = fuzzy_relation  # уже α_i * μ_B_i(y)
                else:
                    # Если impl_type = Мамдани, но comp_type = MAX_PROD, используем product
                    rule_output = truth_level * term_membership
            else:
                rule_output = fuzzy_relation

            # ШАГ 3: Агрегация индивидуальных выходов правил B' = Agg(B'_1, ..., B'_n)
            # Объединяем результаты всех правил согласно типу агрегации
            if agg_type == AggregationType.MAX:
                # MAX агрегация: B'(y) = max_i[B'_i(y)]
                output_mf = np.maximum(output_mf, rule_output)
            elif agg_type == AggregationType.SUM:
                # SUM агрегация: B'(y) = min(1.0, Σ_i[B'_i(y)])
                output_mf = np.minimum(1.0, output_mf + rule_output)
            elif agg_type == AggregationType.PROBOR:
                # PROBOR агрегация: B'(y) = a ⊕ b = a + b - a*b
                output_mf = output_mf + rule_output - output_mf * rule_output
                output_mf = np.minimum(1.0, output_mf)

        return output_mf, x_range
    
    def get_rule_truth_levels(self, inputs: Dict[str, float],
                               output_var: Optional[str] = None) -> List[Tuple[FuzzyRule, float]]:
        """
        Возвращает уровни истинности предпосылок для правил.
        
        Args:
            inputs: входные значения
            output_var: имя выходной переменной для фильтрации правил (опционально)
        
        Returns:
            список кортежей (правило, уровень истинности)
        """
        truth_levels = []
        for rule in self.rules:
            if output_var and rule.result_var != output_var:
                continue
            level = self._evaluate_rule_conditions(rule, inputs)
            truth_levels.append((rule, level))
        return truth_levels
    
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

