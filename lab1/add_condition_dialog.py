from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QLineEdit, QPushButton, QWidget
)

class AddConditionDialog(QDialog):
    def __init__(self, available_data, parent=None):
        super().__init__(parent)
        self.available_data = available_data

        # исключаем рекомендацию
        objects = [obj for obj in available_data.keys() if obj != "рекомендация"]

        self.object_combo = QComboBox()
        self.object_combo.addItems(sorted(objects))
        self.operator_combo = QComboBox()

        # контейнер для значения
        self.value_container = QWidget()
        self.value_layout = QVBoxLayout()
        self.value_layout.setContentsMargins(0,0,0,0)
        self.value_container.setLayout(self.value_layout)
        self.value_edit = None
        self.value_combo = None

        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Отмена")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Объект:"))
        layout.addWidget(self.object_combo)
        layout.addWidget(QLabel("Оператор:"))
        layout.addWidget(self.operator_combo)
        layout.addWidget(QLabel("Значение:"))
        layout.addWidget(self.value_container)
        btns = QHBoxLayout()
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)
        self.setLayout(layout)

        self.object_combo.currentTextChanged.connect(self.update_values_and_operators)
        self.update_values_and_operators(self.object_combo.currentText())

    def update_values_and_operators(self, object_name: str):
        # очищаем контейнер
        for i in reversed(range(self.value_layout.count())):
            widget = self.value_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        if object_name.startswith("количество"):
            self.operator_combo.clear()
            self.operator_combo.addItems([">=", "<=", "=", ">", "<"])
            self.value_edit = QLineEdit()
            self.value_edit.setPlaceholderText("Введите число")
            self.value_combo = None
            self.value_layout.addWidget(self.value_edit)
        else:
            self.operator_combo.clear()
            self.operator_combo.addItem("=")
            self.value_combo = QComboBox()
            self.value_combo.addItems(sorted(self.available_data[object_name]))
            self.value_edit = None
            self.value_layout.addWidget(self.value_combo)

    def get_condition(self):
        obj = self.object_combo.currentText()
        op = self.operator_combo.currentText()
        if self.value_edit:
            val = self.value_edit.text().strip()
        else:
            val = self.value_combo.currentText()
        return obj, op, val

