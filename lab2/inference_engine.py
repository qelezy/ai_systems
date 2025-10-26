from typing import List, Dict, Any, Union
from rule_parser import Rule, ComparisonOperator, LogicalOperator

class InferenceEngine:
    """Движок обратного логического вывода"""

    def __init__(self, rules: List[Rule], facts: Dict[str, Any]):
        self.rules = rules
        self.facts = facts.copy()  # Создаем копию, чтобы не изменять исходные факты
        self.proven_goals = set()  # Кэш доказанных целей
        self.inference_trace = []  # Трассировка вывода

    def prove(self, goal_object: str, goal_value: str, parent_rule_index: int = 0) -> bool:
        """Пытается доказать цель: object=value"""
        goal_key = f"{goal_object}={goal_value}"

        if goal_key in self.proven_goals:
            return True

        # Проверяем, есть ли уже такой факт в базе знаний
        if self.facts.get(goal_object) == goal_value:
            self.proven_goals.add(goal_key)
            self.inference_trace.append(f"Факт уже известен: {goal_object} = {goal_value}")
            return True

        # Ищем правила, ведущие к цели
        for rule_index, rule in enumerate(self.rules, 1):
            if rule.result_object == goal_object and rule.result_value == goal_value:
                self.inference_trace.append(f"Проверяем правило {rule_index}: {rule}")
                if self._evaluate_conditions(rule, rule_index):
                    # Добавляем новый факт в базу знаний
                    self.facts[goal_object] = goal_value
                    self.proven_goals.add(goal_key)
                    self.inference_trace.append(f"Правило {rule_index} выполнено! Добавлен факт: {goal_object} = {goal_value}")
                    return True

        return False

    def _evaluate_conditions(self, rule: Rule, rule_index: int) -> bool:
        """Проверяет выполнение условий"""
        results = []

        for cond in rule.conditions:
            fact_value = self.facts.get(cond.object_name)

            if fact_value is None:
                # Если факт неизвестен — пробуем доказать как новую цель
                self.inference_trace.append(f"Неизвестен факт '{cond.object_name}' для правила {rule_index}, пытаемся доказать: {cond.object_name} = {cond.value}")
                if self.prove(cond.object_name, cond.value, rule_index):
                    fact_value = self.facts.get(cond.object_name)
                    self.inference_trace.append(f"Успешно доказан факт для правила {rule_index}: {cond.object_name} = {fact_value}")
                else:
                    self.inference_trace.append(f"Не удалось доказать факт для правила {rule_index}: {cond.object_name} = {cond.value}")
                    return False

            if fact_value is None or not self._compare(fact_value, cond.operator, cond.value):
                self.inference_trace.append(f"Условие правила {rule_index} не выполнено: {cond.object_name} {cond.operator.value} {cond.value} (факт: {fact_value})")
                results.append(False)
            else:
                self.inference_trace.append(f"Условие правила {rule_index} выполнено: {cond.object_name} {cond.operator.value} {cond.value}")
                results.append(True)

        # Обрабатываем логические операторы (И / ИЛИ)
        result = results[0]
        for op, next_val in zip(rule.logical_operators, results[1:]):
            if op == LogicalOperator.AND:
                result = result and next_val
            elif op == LogicalOperator.OR:
                result = result or next_val

        return result

    def get_inference_trace(self) -> List[str]:
        """Возвращает трассировку процесса вывода"""
        return self.inference_trace.copy()

    def get_facts(self) -> Dict[str, Any]:
        """Возвращает текущее состояние базы знаний"""
        return self.facts.copy()

    def _compare(self, left: Union[int, float, str], operator: ComparisonOperator, right: str) -> bool:
        """Сравнение по оператору"""
        try:
            left_val = float(left)
            right_val = float(right)
            # Если оба значения числовые, сравниваем как числа
            if operator == ComparisonOperator.EQUALS:
                return left_val == right_val
            elif operator == ComparisonOperator.GREATER_EQUAL:
                return left_val >= right_val
            elif operator == ComparisonOperator.LESS_EQUAL:
                return left_val <= right_val
            elif operator == ComparisonOperator.GREATER:
                return left_val > right_val
            elif operator == ComparisonOperator.LESS:
                return left_val < right_val
        except ValueError:
            # Если не удалось преобразовать в числа, сравниваем как строки
            left_val = str(left)
            right_val = str(right)
            if operator == ComparisonOperator.EQUALS:
                return left_val == right_val
            # Для строковых значений другие операторы не поддерживаются
            return False
        return False
