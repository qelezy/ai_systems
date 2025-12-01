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
        """Применяет импликацию для скаляров"""
        if impl_type == ImplicationType.MAMDANI:
            return min(a, b)
        elif impl_type == ImplicationType.LARSEN:
            return a * b
    
    def _implication_matrix(self, a: np.ndarray, b: np.ndarray, impl_type: ImplicationType) -> np.ndarray:
        """Применяет импликацию поэлементно для матриц/векторов"""
        if impl_type == ImplicationType.MAMDANI:
            return np.minimum(a, b)
        elif impl_type == ImplicationType.LARSEN:
            return a * b
        else:
            return b
    
    def _aggregate(self, values: List[float], agg_type: AggregationType) -> float:
        """Применяет оператор агрегации к списку скаляров"""
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
    
    def _aggregate_arrays(self, arrays: np.ndarray, agg_type: AggregationType) -> np.ndarray:
        """Применяет оператор агрегации к массивам (первая ось - агрегация)"""
        if arrays.size == 0:
            return np.array([])
        
        if agg_type == AggregationType.MAX:
            return np.max(arrays, axis=0)
        elif agg_type == AggregationType.SUM:
            return np.minimum(1.0, np.sum(arrays, axis=0))
        elif agg_type == AggregationType.PROBOR:
            result = arrays[0]
            for i in range(1, len(arrays)):
                result = result + arrays[i] - result * arrays[i]
            return np.minimum(1.0, result)
        else:
            return np.max(arrays, axis=0)
    
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

        output_variable = self.variables.get(output_var)
        if not output_variable:
            raise ValueError(f"Переменная {output_var} не найдена")

        x_range = np.linspace(output_variable.min_val, output_variable.max_val, resolution)
        output_mf = np.zeros(resolution)
        
        # Сохраняем все B'_i для явной агрегации на шаге 3
        individual_outputs = []

        # Обрабатываем каждое правило
        for rule in self.rules:
            if rule.result_var != output_var:
                continue

            # ===== ШАГ 1: Вычисление нечётких соответствий R_i(y) = A_i(x) → B_i(y) =====

            
            # Получаем входной терм A_i из условия правила
            input_var_name = list(rule.conditions.keys())[0]  # Предполагаем одну входную переменную
            input_term_name = rule.conditions[input_var_name]
            
            input_variable = self.variables.get(input_var_name)
            if not input_variable:
                continue
                
            input_term_mf = input_variable.terms.get(input_term_name)
            if not input_term_mf:
                continue
            
            # Получаем выходной терм B_i
            result_term_mf = output_variable.terms.get(rule.result_term)
            if not result_term_mf:
                continue
            
            # Строим диапазон для входной переменной (для вычисления A_i(x))
            x_input_range = np.linspace(input_variable.min_val, input_variable.max_val, 
                                        self.condition_resolution)
            
            # Вычисляем A_i(x) для всех x в диапазоне входа
            mu_a_i = np.array([input_term_mf.membership(x) for x in x_input_range])
            
            # Вычисляем B_i(y) для всех y в диапазоне выхода
            mu_b_i = np.array([result_term_mf.membership(y) for y in x_range])
            
            # ШАГ 1: Вычисление нечётких соответствий R_i(x,y) = A_i(x) ⊙ B_i(y)
            # Строим матрицу R_i размером (len(x_input_range), len(x_range))
            # R_i[i,j] = A_i(x_i) ⊙ B_i(y_j)
            
            
            # Вычисляем матрицу импликаций через векторизованную операцию
            R_i_matrix = self._implication_matrix(mu_a_i[:, np.newaxis], mu_b_i[np.newaxis, :], 
                                                  impl_type)

            # ШАГ 2: Вычисление выходов для каждого правила B'_i(y) = A' ○ R_i
            # где A' - фаззифицированный входной синглтон (для входного значения из inputs)
            # ○ - операция композиции (max-min или max-product, определяется comp_type)
            
            # Получаем A'(x) - фаззифицированное входное значение
            x_input_range_crisp, a_prime = self._build_input_membership(input_variable, 
                                                                         inputs[input_var_name])
            
            # Интерполируем A' на сетку x_input_range для согласования размеров
            a_prime_interp = np.interp(x_input_range, x_input_range_crisp, a_prime)
            
            # Вычисляем B'_i(y) = max_x[A'(x) ⊙ R_i(x,y)]
            # где ⊙ определяется типом композиции
            
            if comp_type == CompositionType.MAX_MIN:
                # MAX_MIN композиция: B'_i(y) = max_x[min(A'(x), R_i(x,y))]
                # Для каждого y: берём минимум A'(x) с каждой строкой R_i, потом max по x
                B_i_prime = np.max(np.minimum(a_prime_interp[:, np.newaxis], R_i_matrix), axis=0)
            elif comp_type == CompositionType.MAX_PROD:
                # MAX_PROD композиция: B'_i(y) = max_x[A'(x) * R_i(x,y)]
                # Для каждого y: берём произведение A'(x) с каждой строкой R_i, потом max по x
                B_i_prime = np.max(a_prime_interp[:, np.newaxis] * R_i_matrix, axis=0)
            else:
                B_i_prime = np.max(R_i_matrix, axis=0)

            individual_outputs.append(B_i_prime)

        # ===== ШАГ 3: Агрегация индивидуальных выходов B' = Agg(B'_1, ..., B'_n) =====
        
        if not individual_outputs:
            return output_mf, x_range

        individual_outputs = np.array(individual_outputs)
        
        # Используем векторизованную функцию агрегации
        output_mf = self._aggregate_arrays(individual_outputs, agg_type)

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

