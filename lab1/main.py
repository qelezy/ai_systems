import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QHeaderView
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QColor
from rule_parser import RuleParser


class RulesTableModel(QAbstractTableModel):
    """Модель данных для отображения правил в таблице"""
    
    def __init__(self, rules=None):
        super().__init__()
        self.rules = rules or []
        self.headers = ["№", "Условия", "Логические операторы", "Результат"]
    
    def rowCount(self, parent=QModelIndex()):
        return len(self.rules)
    
    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)
    
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.rules):
            return None
        
        rule = self.rules[index.row()]
        
        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:  # Номер
                return str(index.row() + 1)
            elif index.column() == 1:  # Условия
                return self.format_conditions(rule.conditions)
            elif index.column() == 2:  # Логические операторы
                return self.format_logical_operators(rule.logical_operators)
            elif index.column() == 3:  # Результат
                return f"{rule.result_object} = {rule.result_value}"
        
        return None
    
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return None
    
    def format_conditions(self, conditions):
        """Форматирует условия для отображения в таблице"""
        if not conditions:
            return ""
        
        formatted = []
        for condition in conditions:
            formatted.append(f"{condition.object_name} {condition.operator.value} {condition.value}")
        
        return "\n".join(formatted)
    
    def format_logical_operators(self, operators):
        """Форматирует логические операторы для отображения в таблице"""
        if not operators:
            return "Нет"
        
        return " ".join([op.value for op in operators])
    
    def update_rules(self, rules):
        """Обновляет правила в модели"""
        self.beginResetModel()
        self.rules = rules
        self.endResetModel()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.load_ui()
    
    def load_ui(self):
        """Загружает UI файл и настраивает интерфейс"""
        ui_file = QFile("mainwindow.ui")
        if not ui_file.open(QFile.ReadOnly):
            print(f"Ошибка: Не удалось открыть файл {ui_file.fileName()}")
            return
        
        loader = QUiLoader()
        self.ui = loader.load(ui_file, self)
        ui_file.close()
        
        if self.ui is None:
            print("Ошибка: Не удалось загрузить UI файл")
            return
        
        # Устанавливаем загруженный UI как центральный виджет
        self.setCentralWidget(self.ui.centralwidget)
        
        # Устанавливаем заголовок окна
        self.setWindowTitle("Система подбора предметов для игры League of Legends")
        
        # Устанавливаем размер окна
        self.resize(800, 600)
        
        # Инициализируем парсер правил
        self.rule_parser = RuleParser()
        
        # Загружаем правила из файла
        self.rules = self.rule_parser.parse_rules_from_file("rules.txt")
        print(f"Загружено {len(self.rules)} правил из файла rules.txt")
        
        # Тестируем загруженные правила
        self.test_loaded_rules()
        
        # Настраиваем интерфейс
        self.setup_ui_connections()
        self.setup_rules_table()

    def test_loaded_rules(self):
        """Тестирует загруженные правила из файла rules.txt"""
        print("\n" + "=" * 60)
        print("ТЕСТИРОВАНИЕ ЗАГРУЖЕННЫХ ПРАВИЛ ИЗ RULES.TXT")
        print("=" * 60)
        
        if not self.rules:
            print("❌ Нет загруженных правил для тестирования")
            return
        
        print(f"✅ Загружено {len(self.rules)} правил")
        print("\nДетальная проверка каждого правила:")
        print("-" * 50)
        
        for i, rule in enumerate(self.rules, 1):
            print(f"\n📋 Правило {i}:")
            print(f"   Полный текст: {rule}")
            
            # Проверяем условия
            print(f"   Условия ({len(rule.conditions)}):")
            for j, condition in enumerate(rule.conditions, 1):
                print(f"     {j}. {condition.object_name} {condition.operator.value} {condition.value}")
            
            # Проверяем логические операторы
            if rule.logical_operators:
                print(f"   Логические операторы ({len(rule.logical_operators)}):")
                for j, op in enumerate(rule.logical_operators, 1):
                    print(f"     {j}. {op.value}")
            else:
                print("   Логические операторы: нет")
            
            # Проверяем результат
            print(f"   Результат:")
            print(f"     Объект: '{rule.result_object}'")
            print(f"     Значение: '{rule.result_value}'")
            
            # Проверяем корректность парсинга
            self.validate_rule_parsing(rule, i)
        
        print("\n" + "=" * 60)
        print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("=" * 60)

    def validate_rule_parsing(self, rule, rule_number):
        """Проверяет корректность парсинга конкретного правила"""
        print(f"   🔍 Проверка корректности парсинга:")
        
        # Проверяем, что объект результата не пустой
        if not rule.result_object or rule.result_object.strip() == "":
            print(f"     ❌ ОШИБКА: Пустой объект результата")
        else:
            print(f"     ✅ Объект результата корректен: '{rule.result_object}'")
        
        # Проверяем, что значение результата не пустое
        if not rule.result_value or rule.result_value.strip() == "":
            print(f"     ❌ ОШИБКА: Пустое значение результата")
        else:
            print(f"     ✅ Значение результата корректно: '{rule.result_value}'")
        
        # Проверяем условия
        for j, condition in enumerate(rule.conditions, 1):
            if not condition.object_name or condition.object_name.strip() == "":
                print(f"     ❌ ОШИБКА: Пустое имя объекта в условии {j}")
            elif not condition.value or condition.value.strip() == "":
                print(f"     ❌ ОШИБКА: Пустое значение в условии {j}")
            else:
                print(f"     ✅ Условие {j} корректно: {condition.object_name} {condition.operator.value} {condition.value}")
        
        # Проверяем соответствие количества условий и логических операторов
        expected_operators = len(rule.conditions) - 1
        actual_operators = len(rule.logical_operators)
        if expected_operators != actual_operators:
            print(f"     ❌ ОШИБКА: Несоответствие количества операторов (ожидается {expected_operators}, найдено {actual_operators})")
        else:
            print(f"     ✅ Количество логических операторов корректно: {actual_operators}")

    def setup_ui_connections(self):
        """Настраивает соединения сигналов и слотов для UI элементов"""
        # Подключаем кнопку загрузки правил
        self.ui.loadRulesBtn.clicked.connect(self.load_rules_from_file)
        
        # Подключаем другие кнопки (пока заглушки)
        self.ui.addRuleBtn.clicked.connect(self.add_rule)
        self.ui.deleteRuleBtn.clicked.connect(self.delete_rule)
        self.ui.saveRulesBtn.clicked.connect(self.save_rules)
        self.ui.recommendItemBtn.clicked.connect(self.recommend_item)

    def setup_rules_table(self):
        """Настраивает таблицу для отображения правил"""
        # Создаем модель данных
        self.rules_model = RulesTableModel(self.rules)
        
        # Устанавливаем модель для таблицы
        self.ui.rulesBaseTable.setModel(self.rules_model)
        
        # Настраиваем растягивание колонок
        header = self.ui.rulesBaseTable.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # Номер - фиксированная ширина
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Условия - растягивается
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)  # Операторы - фиксированная
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)  # Результат - растягивается
        
        # Устанавливаем ширину фиксированных колонок
        self.ui.rulesBaseTable.setColumnWidth(0, 50)  # Номер
        self.ui.rulesBaseTable.setColumnWidth(2, 150)  # Логические операторы
        
        # Настраиваем высоту строк для многострочного текста
        self.ui.rulesBaseTable.verticalHeader().setDefaultSectionSize(60)

    def display_rules_in_table(self):
        """Отображает правила в таблице"""
        if hasattr(self, 'rules_model'):
            self.rules_model.update_rules(self.rules)

    def load_rules_from_file(self):
        """Загружает правила из выбранного файла"""
        # Открываем диалог выбора файла
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл с правилами",
            "",
            "Текстовые файлы (*.txt);;Все файлы (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Загружаем правила из файла
            self.rules = self.rule_parser.parse_rules_from_file(file_path)
            
            # Обновляем таблицу
            self.display_rules_in_table()
            
            # Показываем сообщение об успехе
            QMessageBox.information(
                self,
                "Успех",
                f"Загружено {len(self.rules)} правил из файла:\n{file_path}"
            )
            
            # Выводим информацию в консоль
            print(f"\nЗагружено {len(self.rules)} правил из файла: {file_path}")
            
        except Exception as e:
            # Показываем сообщение об ошибке
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось загрузить правила из файла:\n{str(e)}"
            )
            print(f"Ошибка при загрузке файла: {e}")

    def add_rule(self):
        """Добавляет новое правило (заглушка)"""
        QMessageBox.information(self, "Информация", "Функция добавления правила будет реализована позже")

    def delete_rule(self):
        """Удаляет выбранное правило (заглушка)"""
        QMessageBox.information(self, "Информация", "Функция удаления правила будет реализована позже")

    def save_rules(self):
        """Сохраняет правила в файл (заглушка)"""
        QMessageBox.information(self, "Информация", "Функция сохранения правил будет реализована позже")

    def recommend_item(self):
        """Выдает рекомендацию предмета (заглушка)"""
        QMessageBox.information(self, "Информация", "Функция рекомендации предмета будет реализована позже")

def main():
    """Главная функция приложения"""
    app = QApplication(sys.argv)
    
    # Создаем и показываем главное окно
    window = MainWindow()
    window.show()
    
    # Запускаем цикл обработки событий
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
