"""
Парсер для нечётких правил
"""
import re
from typing import List, Dict
from fuzzy_inference_engine import FuzzyRule


class FuzzyRuleParser:
    """Парсер для правил нечёткой логики"""
    
    def parse_rule(self, rule_text: str) -> FuzzyRule:
        """
        Парсит правило вида: ЕСЛИ var1=term1 И var2=term2 ТО var3=term3
        """
        rule_text = rule_text.strip()
        
        if not rule_text.upper().startswith('ЕСЛИ'):
            raise ValueError("Правило должно начинаться с 'ЕСЛИ'")
        
        # Убираем "ЕСЛИ"
        rule_text = rule_text[4:].strip()
        
        # Находим "ТО"
        to_index = rule_text.upper().find(' ТО ')
        if to_index == -1:
            raise ValueError("Правило должно содержать 'ТО'")
        
        conditions_part = rule_text[:to_index].strip()
        result_part = rule_text[to_index + 4:].strip()
        
        # Парсим условия
        conditions = self._parse_conditions(conditions_part)
        
        # Парсим результат
        result_var, result_term = self._parse_result(result_part)
        
        return FuzzyRule(conditions, result_var, result_term)
    
    def _parse_conditions(self, conditions_text: str) -> Dict[str, str]:
        """Парсит условия вида: var1=term1 И var2=term2"""
        conditions = {}
        
        # Разделяем по "И" или "ИЛИ"
        parts = re.split(r'\s+(И|ИЛИ)\s+', conditions_text, flags=re.IGNORECASE)
        
        # Обрабатываем каждую часть
        for part in parts:
            if part.upper() in ['И', 'ИЛИ']:
                continue
            
            # Парсим условие вида var=term
            match = re.match(r'(\w+)\s*=\s*(\w+)', part.strip())
            if match:
                var_name = match.group(1).strip()
                term_name = match.group(2).strip()
                conditions[var_name] = term_name
        
        return conditions
    
    def _parse_result(self, result_text: str) -> tuple[str, str]:
        """Парсит результат вида: var=term"""
        match = re.match(r'(\w+)\s*=\s*(\w+)', result_text.strip())
        if not match:
            raise ValueError("Неверный формат результата. Ожидается: переменная=терм")
        
        var_name = match.group(1).strip()
        term_name = match.group(2).strip()
        
        return var_name, term_name
    
    def parse_rules_from_file(self, file_path: str) -> List[FuzzyRule]:
        """Парсит правила из файла"""
        rules = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                try:
                    rule = self.parse_rule(line)
                    rules.append(rule)
                except ValueError as e:
                    print(f"Ошибка в строке {line_num}: {e}")
                    print(f"Строка: {line}")
                    continue
        
        return rules

