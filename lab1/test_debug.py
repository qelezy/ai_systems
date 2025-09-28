#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой тест для отладки парсера
"""

from rule_parser import RuleParser


def test_debug():
    """Тестирует парсер с отладочной информацией"""
    parser = RuleParser()
    
    print("=" * 60)
    print("ОТЛАДОЧНЫЙ ТЕСТ ПАРСЕРА")
    print("=" * 60)
    
    # Простое правило для тестирования
    rule_text = "ЕСЛИ мана<30 ТО предмет=зелье_маны"
    
    print(f"Тестируем правило: {rule_text}")
    print("-" * 40)
    
    try:
        rule = parser.parse_rule(rule_text)
        print(f"✓ Успешно распарсено: {rule}")
        print(f"  Объект результата: '{rule.result_object}'")
        print(f"  Значение результата: '{rule.result_value}'")
    except ValueError as e:
        print(f"✗ Ошибка: {e}")


if __name__ == "__main__":
    test_debug()
