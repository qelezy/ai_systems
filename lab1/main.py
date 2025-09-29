import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QHeaderView, QInputDialog, QDialog
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QAbstractTableModel, QModelIndex
from rule_parser import RuleParser, Rule, Condition, ComparisonOperator
from add_condition_dialog import AddConditionDialog
import re

class RulesTableModel(QAbstractTableModel):
    """Модель данных для отображения правил в таблице"""
    
    def __init__(self, rules=None):
        super().__init__()
        self.rules = rules or []
        self.headers = ["№", "Правило"]
    
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
            elif index.column() == 1:  # Правило
                return str(rule)
        
        return None
    
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return None
    
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
        self.rules = []
        # self.rule_parser.parse_rules_from_file("rules.txt")
        # print(f"Загружено {len(self.rules)} правил из файла rules.txt")
        
        # Настраиваем интерфейс
        self.setup_ui_connections()
        self.setup_rules_table()

    def setup_ui_connections(self):
        """Настраивает соединения сигналов и слотов для UI элементов"""
        self.ui.addRuleBtn.clicked.connect(self.add_rule)
        self.ui.deleteRuleBtn.clicked.connect(self.delete_rule)
        self.ui.saveRulesBtn.clicked.connect(self.save_rules)
        self.ui.loadRulesBtn.clicked.connect(self.load_rules_from_file)
        self.ui.addConditionBtn.clicked.connect(self.add_condition)
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
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Правило - растягивается
        
        # Устанавливаем ширину фиксированных колонок
        self.ui.rulesBaseTable.setColumnWidth(0, 30)  # Номер
        
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
            self.available_data = self.rule_parser.extract_objects_and_values(self.rules)
            
            # Обновляем таблицу
            self.display_rules_in_table()
            
            # Показываем сообщение об успехе
            QMessageBox.information(
                self,
                "Успех",
                f"Загружено {len(self.rules)} правил из файла:\n{file_path}"
            )
            
        except Exception as e:
            # Показываем сообщение об ошибке
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось загрузить правила из файла:\n{str(e)}"
            )
            print(f"Ошибка при загрузке файла: {e}")

    def add_rule(self):
        """Добавляет новое правило"""
        text, ok = QInputDialog.getText(
            self,
            "Добавить правило",
            "Введите правило в формате:\nЕСЛИ ... ТО ..."
        )
        if ok and text.strip():
            try:
                # Парсим одно правило
                rule = self.rule_parser.parse_rule(text.strip())
                self.rules.append(rule)
                self.display_rules_in_table()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Неверный формат правила:\n{e}")

    def delete_rule(self):
        """Удаляет выбранное правило"""
        selection = self.ui.rulesBaseTable.selectionModel().selectedRows()
        if not selection:
            QMessageBox.warning(self, "Удаление", "Выберите правило для удаления")
            return

        row = selection[0].row()
        rule = self.rules[row]

        confirm = QMessageBox.question(
            self,
            "Подтверждение удаления",
            f"Удалить правило?\n\n{rule}",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            self.rules.pop(row)
            self.display_rules_in_table()

    def save_rules(self):
        """Сохраняет правила в файл"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить правила",
            "rules.txt",
            "Текстовые файлы (*.txt);;Все файлы (*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for rule in self.rules:
                    f.write(str(rule) + "\n")
            QMessageBox.information(self, "Сохранение", f"Правила сохранены в файл:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить правила:\n{e}")

    def add_condition(self):
        # открываем диалог
        dlg = AddConditionDialog(self.available_data, self)
        if dlg.exec():  # если нажали OK
            obj, op, val = dlg.get_condition()
            condition_text = f"{obj}{op}{val}"
            
            # добавляем в QListWidget
            self.ui.currentConditions.addItem(condition_text)

    def recommend_item(self):
        """Выдает рекомендацию предмета на основе выбранных условий"""
        # Собираем выбранные условия из QListWidget
        selected_conditions = {}
        for i in range(self.ui.currentConditions.count()):
            item_text = self.ui.currentConditions.item(i).text()  # пример: "количество_врагов_ап>=3"
            
            # Разбираем строку на объект, оператор и значение
            match = re.match(r'(.+?)(>=|<=|=|>|<)(.+)', item_text)
            if not match:
                continue
            obj, op, val = match.groups()
            selected_conditions[obj.strip()] = (op, val.strip())

        # Пробегаем все правила и ищем подходящее
        recommendation = None
        for rule in self.rules:
            match_rule = True
            for cond, op in zip(rule.conditions, rule.logical_operators + [None]):  # последний оператор None
                if cond.object_name not in selected_conditions:
                    match_rule = False
                    break

                user_op, user_val = selected_conditions[cond.object_name]
                # преобразуем числа, если это количество
                try:
                    cond_val = int(cond.value)
                    user_val_int = int(user_val)
                    if cond.operator == ComparisonOperator.GREATER_EQUAL and not (user_val_int >= cond_val):
                        match_rule = False
                    elif cond.operator == ComparisonOperator.LESS_EQUAL and not (user_val_int <= cond_val):
                        match_rule = False
                    elif cond.operator == ComparisonOperator.EQUALS and not (user_val == cond.value):
                        match_rule = False
                    elif cond.operator == ComparisonOperator.GREATER and not (user_val_int > cond_val):
                        match_rule = False
                    elif cond.operator == ComparisonOperator.LESS and not (user_val_int < cond_val):
                        match_rule = False
                except ValueError:
                    # не число, сравниваем строки
                    if cond.operator == ComparisonOperator.EQUALS and not (user_val == cond.value):
                        match_rule = False
                    elif cond.operator != ComparisonOperator.EQUALS:
                        # для текстовых полей только "=" поддерживаем
                        match_rule = False

                if not match_rule:
                    break

            if match_rule:
                recommendation = f"{rule.result_object} = {rule.result_value}"
                break  # остановились на первом совпадении

        # Выводим результат в QTextEdit
        if recommendation:
            self.ui.result.setPlainText(recommendation)
        else:
            self.ui.result.setPlainText("Подходящее правило не найдено")


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
