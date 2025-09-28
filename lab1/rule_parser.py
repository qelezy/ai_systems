import re
from typing import List, Dict, Any, Union
from enum import Enum


class ComparisonOperator(Enum):
    """Операторы сравнения"""
    EQUALS = "="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    GREATER = ">"
    LESS = "<"


class LogicalOperator(Enum):
    """Логические операторы"""
    AND = "И"
    OR = "ИЛИ"


class Condition:
    """Представляет одно условие в правиле"""
    
    def __init__(self, object_name: str, operator: ComparisonOperator, value: str):
        self.object_name = object_name.strip()
        self.operator = operator
        self.value = value.strip()
    
    def __str__(self):
        return f"{self.object_name}{self.operator.value}{self.value}"
    
    def __repr__(self):
        return f"Condition({self.object_name}, {self.operator}, {self.value})"


class Rule:
    """Представляет правило экспертной системы"""
    
    def __init__(self, conditions: List[Condition], logical_operators: List[LogicalOperator], 
                 result_object: str, result_value: str):
        self.conditions = conditions
        self.logical_operators = logical_operators
        self.result_object = result_object.strip()
        self.result_value = result_value.strip()
    
    def __str__(self):
        conditions_str = str(self.conditions[0])
        for i, (op, cond) in enumerate(zip(self.logical_operators, self.conditions[1:]), 1):
            conditions_str += f" {op.value} {cond}"
        
        return f"ЕСЛИ {conditions_str} ТО {self.result_object}={self.result_value}"
    
    def __repr__(self):
        return f"Rule({self.conditions}, {self.logical_operators}, {self.result_object}, {self.result_value})"


class RuleParser:
    """Парсер для правил экспертной системы"""
    
    def __init__(self):
        # Паттерн для поиска условий с различными операторами
        self.condition_pattern = r'(\w+)\s*(>=|<=|>|<|=)\s*([^ИИЛИ\s]+)'
        # Паттерн для поиска логических операторов
        self.logical_pattern = r'\s+(И|ИЛИ)\s+'
        # Паттерн для поиска результата (объект может содержать знаки сравнения)
        self.result_pattern = r'([^\s=]+)\s*=\s*(.+)$'
    
    def parse_rule(self, rule_text: str) -> Rule:
        """
        Парсит текстовое правило и возвращает объект Rule
        
        Args:
            rule_text: Текст правила вида "ЕСЛИ объект1=значение1 И объект2>=значение2 ТО объект3=значение3"
        
        Returns:
            Rule: Объект правила
            
        Raises:
            ValueError: Если правило не может быть распарсено
        """
        rule_text = rule_text.strip()
        
        # Проверяем, что правило начинается с "ЕСЛИ"
        if not rule_text.upper().startswith('ЕСЛИ'):
            raise ValueError("Правило должно начинаться с 'ЕСЛИ'")
        
        # Убираем "ЕСЛИ" из начала
        rule_text = rule_text[4:].strip()
        
        # Находим позицию "ТО"
        to_index = rule_text.upper().find(' ТО ')
        if to_index == -1:
            raise ValueError("Правило должно содержать 'ТО'")
        
        # Разделяем на условия и результат
        conditions_part = rule_text[:to_index].strip()
        result_part = rule_text[to_index + 4:].strip()
        
        # Парсим условия
        conditions, logical_operators = self._parse_conditions(conditions_part)
        
        # Парсим результат
        result_object, result_value = self._parse_result(result_part)
        
        return Rule(conditions, logical_operators, result_object, result_value)
    
    def _parse_conditions(self, conditions_text: str) -> tuple[List[Condition], List[LogicalOperator]]:
        """Парсит часть с условиями"""
        conditions = []
        logical_operators = []
        
        # Находим все условия
        condition_matches = re.findall(self.condition_pattern, conditions_text)
        
        if not condition_matches:
            raise ValueError("Не найдено ни одного условия")
        
        # Создаем объекты условий
        for obj_name, op_str, value in condition_matches:
            operator = self._get_operator(op_str)
            condition = Condition(obj_name, operator, value)
            conditions.append(condition)
        
        # Находим логические операторы
        logical_matches = re.findall(self.logical_pattern, conditions_text)
        
        for op_str in logical_matches:
            if op_str == "И":
                logical_operators.append(LogicalOperator.AND)
            elif op_str == "ИЛИ":
                logical_operators.append(LogicalOperator.OR)
            else:
                raise ValueError(f"Неизвестный логический оператор: {op_str}")
        
        # Проверяем соответствие количества условий и операторов
        if len(conditions) != len(logical_operators) + 1:
            raise ValueError("Количество логических операторов должно быть на 1 меньше количества условий")
        
        return conditions, logical_operators
    
    def _parse_result(self, result_text: str) -> tuple[str, str]:
        """Парсит часть с результатом"""
        match = re.match(self.result_pattern, result_text)
        if not match:
            raise ValueError("Неверный формат результата. Ожидается: объект=значение")
        
        # Убираем лишние пробелы из объекта и значения
        object_name = match.group(1).strip()
        value = match.group(2).strip()
        
        return object_name, value
    
    def _get_operator(self, op_str: str) -> ComparisonOperator:
        """Преобразует строку оператора в enum"""
        operator_map = {
            "=": ComparisonOperator.EQUALS,
            ">=": ComparisonOperator.GREATER_EQUAL,
            "<=": ComparisonOperator.LESS_EQUAL,
            ">": ComparisonOperator.GREATER,
            "<": ComparisonOperator.LESS
        }
        
        if op_str not in operator_map:
            raise ValueError(f"Неизвестный оператор сравнения: {op_str}")
        
        return operator_map[op_str]
    
    def parse_rules_from_file(self, file_path: str) -> List[Rule]:
        """
        Парсит правила из файла
        
        Args:
            file_path: Путь к файлу с правилами
            
        Returns:
            List[Rule]: Список правил
        """
        rules = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    
                    # Пропускаем пустые строки и комментарии
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        rule = self.parse_rule(line)
                        rules.append(rule)
                    except ValueError as e:
                        print(f"Ошибка в строке {line_num}: {e}")
                        print(f"Строка: {line}")
                        continue
        
        except FileNotFoundError:
            print(f"Файл {file_path} не найден")
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
        
        return rules
    
    def save_rules_to_file(self, rules: List[Rule], file_path: str):
        """
        Сохраняет правила в файл
        
        Args:
            rules: Список правил для сохранения
            file_path: Путь к файлу для сохранения
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                for rule in rules:
                    file.write(str(rule) + '\n')
        except Exception as e:
            print(f"Ошибка при сохранении файла: {e}")


# Пример использования
if __name__ == "__main__":
    parser = RuleParser()
    
    # Тестовые правила
    test_rules = [
        "ЕСЛИ здоровье<50 ТО предмет=зелье_здоровья",
        "ЕСЛИ урон>100 И защита>=50 ТО предмет=меч_дракона",
        "ЕСЛИ мана<=30 ИЛИ здоровье<=25 ТО предмет=зелье_маны",
        "ЕСЛИ уровень>=10 И золото>=1000 ТО предмет=мифический_предмет"
    ]
    
    print("Тестирование парсера правил:")
    print("=" * 50)
    
    for i, rule_text in enumerate(test_rules, 1):
        try:
            rule = parser.parse_rule(rule_text)
            print(f"Правило {i}: {rule}")
            print(f"  Условия: {rule.conditions}")
            print(f"  Логические операторы: {rule.logical_operators}")
            print(f"  Результат: {rule.result_object}={rule.result_value}")
            print()
        except ValueError as e:
            print(f"Ошибка в правиле {i}: {e}")
            print()
