"""
GUI приложение для системы нечёткого логического вывода
"""
import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QGroupBox,
    QSpinBox, QMessageBox, QFileDialog
)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    # Fallback для старых версий matplotlib
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from fuzzy_sets_parser import FuzzySetsParser
from fuzzy_rule_parser import FuzzyRuleParser
from fuzzy_inference_engine import (
    FuzzyInferenceEngine,
    ImplicationType,
    CompositionType,
    AggregationType
)


class FuzzyPlotWidget(QWidget):
    """Виджет для отображения графиков matplotlib"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def clear(self):
        """Очищает график"""
        self.figure.clear()
        self.canvas.draw()
    
    def plot_inference_result(self, membership, x_range, title="Результат нечёткого вывода"):
        """Отображает результат нечёткого вывода"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.plot(x_range, membership, 'b-', linewidth=2, label='Результат вывода')
        ax.fill_between(x_range, membership, alpha=0.3)
        ax.set_xlabel('Выходная переменная')
        ax.set_ylabel('Степень принадлежности')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        self.canvas.draw()
    
    def plot_comparison(self, x_data, y_data_list, labels, title="Сравнение результатов"):
        """Отображает сравнение нескольких результатов"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        colors = ['b-', 'r--', 'g--', 'm-', 'c-']
        for y_data, label, color in zip(y_data_list, labels, colors):
            ax.plot(x_data, y_data, color, linewidth=2, label=label)
        
        ax.set_xlabel('Входная переменная')
        ax.set_ylabel('Выходная переменная')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()


class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.sets_parser = FuzzySetsParser()
        self.rule_parser = FuzzyRuleParser()
        self.rules = []
        self.engine = None
        self.current_system_type = "single"  # "single", "multiple", "mamdani"
        
        self.load_ui()
        self.load_default_data()
    
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
        self.setWindowTitle("Система нечёткого логического вывода")
        
        # Устанавливаем размер окна
        self.resize(1400, 800)
        
        # Добавляем виджет для графика в контейнер
        self.plot_widget = FuzzyPlotWidget()
        plot_layout = QVBoxLayout(self.ui.plotWidgetContainer)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.plot_widget)
        
        # Настраиваем интерфейс
        self.setup_ui_connections()
        self.setup_input_widgets()
    
    def setup_ui_connections(self):
        """Настраивает соединения сигналов и слотов для UI элементов"""
        self.ui.systemCombo.currentIndexChanged.connect(self.on_system_changed)
        self.ui.calculateBtn.clicked.connect(self.calculate)
        self.ui.loadBtn.clicked.connect(self.load_data)
        self.ui.compareImplBtn.clicked.connect(self.compare_implications)
        self.ui.systemCombo.currentIndexChanged.connect(self.update_mechanism_visibility)
    
    def setup_input_widgets(self):
        """Инициализирует виджеты ввода"""
        # Сохраняем ссылку на layout для динамического добавления виджетов
        self.input_layout = self.ui.inputLayout
        # Инициализируем первую систему
        self.on_system_changed(0)
    
    def create_single_input_widgets(self):
        """Создаёт виджеты для системы с 1 входом/1 выходом"""
        self.clear_input_widgets()
        
        label = QLabel("Количество предметов (0-6):")
        self.input_layout.addWidget(label)
        
        self.input1_spin = QSpinBox()
        self.input1_spin.setRange(0, 6)
        self.input1_spin.setValue(0)
        self.input1_spin.setSingleStep(1)
        self.input_layout.addWidget(self.input1_spin)
    
    def create_multiple_input_widgets(self):
        """Создаёт виджеты для системы с несколькими входами"""
        self.clear_input_widgets()
        
        labels = [
            "Количество предметов (0-6):",
            "Количество союзников рядом (1-5):",
            "Количество врагов рядом (1-5):"
        ]
        ranges = [(0, 6), (1, 5), (1, 5)]
        self.input_spins = []
        
        for label_text, (min_val, max_val) in zip(labels, ranges):
            label = QLabel(label_text)
            self.input_layout.addWidget(label)
            
            spin = QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(int(min_val))
            spin.setSingleStep(1)
            self.input_spins.append(spin)
            self.input_layout.addWidget(spin)
    
    def create_mamdani_input_widgets(self):
        """Создаёт виджеты для модели Мамдани (с двумя входами)"""
        self.clear_input_widgets()
        
        labels = ["Количество врагов AP (0-5):", "Количество врагов Танк (0-5):"]
        self.input_spins = []
        
        for label_text in labels:
            label = QLabel(label_text)
            self.input_layout.addWidget(label)
            
            spin = QSpinBox()
            spin.setRange(0, 5)
            spin.setValue(2)
            spin.setSingleStep(1)
            self.input_spins.append(spin)
            self.input_layout.addWidget(spin)
    
    def clear_input_widgets(self):
        """Очищает виджеты ввода"""
        while self.input_layout.count():
            item = self.input_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def on_system_changed(self, index):
        """Обработчик изменения типа системы"""
        if index == 0:
            self.current_system_type = "single"
            self.create_single_input_widgets()
        elif index == 1:
            self.current_system_type = "multiple"
            self.create_multiple_input_widgets()
        elif index == 2:
            self.current_system_type = "mamdani"
            self.create_mamdani_input_widgets()
        self.update_mechanism_visibility()
    
    def update_mechanism_visibility(self):
        """Обновляет видимость элементов в зависимости от типа системы"""
        is_single = (self.current_system_type == "single")
        self.ui.mechanismLabel.setVisible(is_single)
        self.ui.mechanismCombo.setVisible(is_single)
        self.ui.compareImplBtn.setVisible(is_single)
    
    def load_default_data(self):
        """Загружает данные по умолчанию"""
        try:
            self.sets_parser.parse_file("fuzzy_sets.txt")
            self.rules = self.rule_parser.parse_rules_from_file("rules.txt")
            self.ui.resultText.append("Данные загружены успешно.")
            self.on_system_changed(0)  # Инициализируем первую систему
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить данные: {e}")
    
    def load_data(self):
        """Загружает данные из файлов"""
        try:
            sets_file, _ = QFileDialog.getOpenFileName(
                self, "Выберите файл с нечёткими множествами", "", "Text files (*.txt)"
            )
            if sets_file:
                self.sets_parser = FuzzySetsParser()
                self.sets_parser.parse_file(sets_file)
            
            rules_file, _ = QFileDialog.getOpenFileName(
                self, "Выберите файл с правилами", "", "Text files (*.txt)"
            )
            if rules_file:
                self.rules = self.rule_parser.parse_rules_from_file(rules_file)
            
            self.ui.resultText.append("Данные загружены успешно.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки: {e}")
    
    def get_implication_type(self):
        """Возвращает выбранный тип импликации"""
        impl_map = {
            0: ImplicationType.MAMDANI,
            1: ImplicationType.LARSEN
        }
        return impl_map[self.ui.implCombo.currentIndex()]
    
    def get_composition_type(self):
        """Возвращает выбранный тип композиции"""
        return CompositionType.MAX_MIN if self.ui.compCombo.currentIndex() == 0 else CompositionType.MAX_PROD
    
    def get_aggregation_type(self):
        """Возвращает выбранный тип агрегации"""
        agg_map = {
            0: AggregationType.MAX,
            1: AggregationType.SUM
        }
        return agg_map[self.ui.aggCombo.currentIndex()]
    
    def calculate(self):
        """Выполняет вычисления нечёткого вывода"""
        try:
            if self.current_system_type == "single":
                self.calculate_single()
            elif self.current_system_type == "multiple":
                self.calculate_multiple()
            elif self.current_system_type == "mamdani":
                self.calculate_mamdani()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка вычисления: {e}")
            import traceback
            traceback.print_exc()
    
    def calculate_single(self):
        """Вычисление для системы 1 вход/1 выход"""
        input_val = self.input1_spin.value()
        inputs = {"количество_предметов": input_val}
        
        # Фильтруем правила
        single_rules = [r for r in self.rules if "количество_предметов" in r.conditions and "текущая_сила" in r.result_var]
        
        if not single_rules:
            QMessageBox.warning(self, "Предупреждение", "Не найдено правил для данной системы")
            return
        
        engine = FuzzyInferenceEngine(single_rules, self.sets_parser.get_all_variables())
        
        # Выбираем механизм вывода
        mechanism = self.ui.mechanismCombo.currentIndex()
        
        if mechanism == 0:  # Уровни истинности
            membership, x_range = engine.inference_truth_level(
                inputs, "текущая_сила",
                impl_type=self.get_implication_type(),
                agg_type=self.get_aggregation_type()
            )
            mechanism_name = "Уровни истинности предпосылок"
        elif mechanism == 1:  # Композиция Max-Min
            membership, x_range = engine.inference_composition(
                inputs, "текущая_сила",
                impl_type=self.get_implication_type(),
                comp_type=CompositionType.MAX_MIN,
                agg_type=self.get_aggregation_type()
            )
            mechanism_name = "Композиция Max-Min"
        else:  # Композиция Max-Product
            membership, x_range = engine.inference_composition(
                inputs, "текущая_сила",
                impl_type=self.get_implication_type(),
                comp_type=CompositionType.MAX_PROD,
                agg_type=self.get_aggregation_type()
            )
            mechanism_name = "Композиция Max-Product"
        
        output = engine.defuzzify_centroid(membership, x_range)
        
        # Отображаем результат
        self.plot_widget.plot_inference_result(membership, x_range, f"Результат нечёткого вывода ({mechanism_name})")
        
        self.ui.resultText.clear()
        self.ui.resultText.append(f"Входное значение: {int(input_val)}")
        self.ui.resultText.append(f"Выходное значение: {output:.2f}")
        self.ui.resultText.append(f"Механизм вывода: {mechanism_name}")
        self.ui.resultText.append(f"Тип импликации: {self.ui.implCombo.currentText()}")
        self.ui.resultText.append(f"Тип агрегации: {self.ui.aggCombo.currentText()}")
    
    def calculate_multiple(self):
        """Вычисление для системы с несколькими входами"""

        # 1. Вычисляем текущая_сила по количеству_предметов
        amount_of_items = int(self.input_spins[0].value())
        inputs_strength = {"количество_предметов": amount_of_items}
        
        rules_strength = [r for r in self.rules if r.result_var == "текущая_сила"]
        if not rules_strength:
            QMessageBox.warning(self, "Предупреждение", "Не найдены правила для вычисления текущей_сила")
            return
        
        engine_strength = FuzzyInferenceEngine(rules_strength, self.sets_parser.get_all_variables())
        membership_strength, x_range_strength = engine_strength.inference_truth_level(
            inputs_strength, "текущая_сила",
            impl_type=self.get_implication_type(),
            agg_type=self.get_aggregation_type()
        )
        current_strength = engine_strength.defuzzify_centroid(membership_strength, x_range_strength)
        
        # 2. Вычисляем готовность_к_бою
        inputs = {
            "текущая_сила": current_strength,
            "количество_союзников_рядом": self.input_spins[1].value(),
            "количество_врагов_рядом": self.input_spins[2].value()
        }
        
        # Фильтруем правила
        multi_rules = [r for r in self.rules if "готовность_к_бою" in r.result_var]
        
        if not multi_rules:
            QMessageBox.warning(self, "Предупреждение", "Не найдено правил для данной системы")
            return
        
        engine = FuzzyInferenceEngine(multi_rules, self.sets_parser.get_all_variables())
        
        membership, x_range = engine.inference_truth_level(
            inputs, "готовность_к_бою",
            impl_type=self.get_implication_type(),
            agg_type=self.get_aggregation_type()
        )
        
        output = engine.defuzzify_centroid(membership, x_range)
        
        # Отображаем результат
        self.plot_widget.plot_inference_result(membership, x_range, "Результат нечёткого вывода")
        
        self.ui.resultText.clear()
        self.ui.resultText.append(f"Входные значения:")
        self.ui.resultText.append(f"  количество_предметов: {amount_of_items}")
        self.ui.resultText.append(f"  текущая_сила (вычислено): {current_strength:.2f}")
        self.ui.resultText.append(f"  количество_союзников_рядом: {int(self.input_spins[1].value())}")
        self.ui.resultText.append(f"  количество_врагов_рядом: {int(self.input_spins[2].value())}")
        self.ui.resultText.append(f"\nВыходное значение: {output:.2f}")
    
    def calculate_mamdani(self):
        """Вычисление для модели Мамдани (с двумя входами)"""
        if len(self.input_spins) < 2:
            QMessageBox.warning(self, "Предупреждение", "Необходимо два входных значения")
            return
        
        x1_val = self.input_spins[0].value()
        x2_val = self.input_spins[1].value()
        inputs = {
            "количество_врагов_ап": x1_val,
            "количество_врагов_танк": x2_val
        }
        
        # Фильтруем правила для модели Мамдани (используем те же переменные, что и в части 1)
        # Ищем правила, которые используют только количество_врагов_ап и количество_врагов_танк (без крит)
        mamdani_rules = [r for r in self.rules if "приоритет_защиты" in r.result_var and 
                         "количество_врагов_ап" in r.conditions and 
                         "количество_врагов_танк" in r.conditions and
                         "количество_врагов_крит" not in r.conditions]
        
        if not mamdani_rules:
            QMessageBox.warning(self, "Предупреждение", "Не найдено правил для модели Мамдани")
            return
        
        engine = FuzzyInferenceEngine(mamdani_rules, self.sets_parser.get_all_variables())
        
        membership, y_range = engine.inference_truth_level(
            inputs, "приоритет_защиты",
            impl_type=self.get_implication_type(),
            agg_type=self.get_aggregation_type()
        )
        
        output = engine.defuzzify_centroid(membership, y_range)
        
        # Вычисляем исходную функцию для сравнения
        x1_norm = x1_val / 5.0
        x2_norm = x2_val / 5.0
        y_original = (15 * x1_norm**2 + 12 * x2_norm**2 + 8 * x1_norm * x2_norm) * 20
        
        # Отображаем результат
        self.plot_widget.plot_inference_result(membership, y_range, "Результат модели Мамдани")
        
        self.ui.resultText.clear()
        self.ui.resultText.append(f"Входные значения:")
        self.ui.resultText.append(f"  Количество врагов AP (x1): {int(x1_val)}")
        self.ui.resultText.append(f"  Количество врагов Танк (x2): {int(x2_val)}")
        self.ui.resultText.append(f"\nВыходное значение (модель Мамдани): {output:.2f}")
        self.ui.resultText.append(f"Исходная функция: y = {y_original:.2f}")
        self.ui.resultText.append(f"Ошибка: {abs(y_original - output):.2f}")
    
    def compare_implications(self):
        """Сравнивает различные типы импликаций для системы 1 вход/1 выход"""
        if self.current_system_type != "single":
            QMessageBox.warning(self, "Предупреждение", "Сравнение импликаций доступно только для системы 1 вход/1 выход")
            return
        
        try:
            input_val = self.input1_spin.value()
            inputs = {"количество_предметов": input_val}
            
            # Фильтруем правила
            single_rules = [r for r in self.rules if "количество_предметов" in r.conditions and "готовность_к_бою" in r.result_var]
            
            if not single_rules:
                QMessageBox.warning(self, "Предупреждение", "Не найдено правил для данной системы")
                return
            
            engine = FuzzyInferenceEngine(single_rules, self.sets_parser.get_all_variables())
            
            # Выбираем механизм вывода
            mechanism = self.ui.mechanismCombo.currentIndex()
            
            implication_types = [
                (ImplicationType.MAMDANI, "Мамдани"),
                (ImplicationType.LARSEN, "Ларсен")
            ]
            
            results = []
            labels = []
            
            for impl_type, impl_name in implication_types:
                if mechanism == 0:  # Уровни истинности
                    membership, x_range = engine.inference_truth_level(
                        inputs, "готовность_к_бою",
                        impl_type=impl_type,
                        agg_type=self.get_aggregation_type()
                    )
                elif mechanism == 1:  # Композиция Max-Min
                    membership, x_range = engine.inference_composition(
                        inputs, "готовность_к_бою",
                        impl_type=impl_type,
                        comp_type=CompositionType.MAX_MIN,
                        agg_type=self.get_aggregation_type()
                    )
                else:  # Композиция Max-Product
                    membership, x_range = engine.inference_composition(
                        inputs, "готовность_к_бою",
                        impl_type=impl_type,
                        comp_type=CompositionType.MAX_PROD,
                        agg_type=self.get_aggregation_type()
                    )
                
                output = engine.defuzzify_centroid(membership, x_range)
                results.append(output)
                labels.append(impl_name)
            
            # Визуализация сравнения
            self.plot_widget.figure.clear()
            ax = self.plot_widget.figure.add_subplot(111)
            
            x_pos = np.arange(len(labels))
            ax.bar(x_pos, results, alpha=0.7)
            ax.set_xlabel('Тип импликации')
            ax.set_ylabel('Выходное значение')
            ax.set_title(f'Сравнение импликаций (вход = {int(input_val)})')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            self.plot_widget.figure.tight_layout()
            self.plot_widget.canvas.draw()
            
            # Вывод результатов
            self.ui.resultText.clear()
            self.ui.resultText.append(f"Сравнение импликаций для входного значения: {int(input_val)}\n")
            for impl_name, result in zip(labels, results):
                self.ui.resultText.append(f"{impl_name:15s}: {result:.2f}")
        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка сравнения: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Главная функция"""
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
