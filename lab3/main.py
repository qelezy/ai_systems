"""
GUI приложение для системы нечёткого логического вывода с выбором механизма
"""
import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QSpinBox, QMessageBox
)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from fuzzy_json_parser import JSONFuzzyModelParser
from fuzzy_inference_engine import (
    FuzzyInferenceEngine,
    ImplicationType,
    AggregationType,
    CompositionType
)

MODEL_FILE = "fuzzy_config.json"


class FuzzyPlotWidget(QWidget):
    """Виджет для отображения графиков matplotlib"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_firing_levels(self, truth_levels, title="Уровни истинности правил"):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if not truth_levels:
            ax.text(0.5, 0.5, "Нет активных правил", ha='center', va='center', fontsize=11)
            ax.set_axis_off()
            ax.set_title(title)
        else:
            indices = np.arange(1, len(truth_levels) + 1)
            values = [level for _, level in truth_levels]
            ax.plot(indices, values, marker='o', linewidth=2, color='#4C72B0')
            ax.fill_between(indices, values, alpha=0.2, color='#4C72B0')
            ax.set_xticks(indices)
            ax.set_xticklabels([str(i) for i in indices])
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('Номер правила')
            ax.set_ylabel('α (уровень истинности)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='y')
        self.canvas.draw()


class MainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.parser = JSONFuzzyModelParser()
        self.variables = {}
        self.rules = []
        self.input_variables = []
        self.output_variable = None
        self.load_default_data()

        self.load_ui()
        self.create_input_widgets()

    def load_ui(self):
        ui_file = QFile("mainwindow.ui")
        if not ui_file.open(QFile.ReadOnly):
            print(f"Ошибка: Не удалось открыть файл {ui_file.fileName()}")
            return

        loader = QUiLoader()
        self.ui = loader.load(ui_file)
        ui_file.close()

        if self.ui is None:
            print("Ошибка: Не удалось загрузить UI файл")
            return

        self.setCentralWidget(self.ui.centralwidget)
        self.setWindowTitle("Система нечёткого логического вывода")
        self.resize(1400, 800)

        # График
        self.plot_widget = FuzzyPlotWidget()
        plot_layout = QVBoxLayout(self.ui.plotWidgetContainer)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.plot_widget)

        # Кнопка
        self.ui.calculateBtn.clicked.connect(self.calculate)

    def load_default_data(self):
        try:
            (
                self.variables,
                self.rules,
                self.input_variables,
                self.output_variable
            ) = self.parser.parse_file(MODEL_FILE)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить данные: {e}")

    def create_input_widgets(self):
        layout = self.ui.inputLayout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Входное значение
        layout.addWidget(QLabel("Количество предметов (0-6):"))
        self.input1_spin = QSpinBox()
        self.input1_spin.setRange(0, 6)
        self.input1_spin.setValue(0)
        layout.addWidget(self.input1_spin)

        # Механизм логического вывода
        self.mechanismCombo = self.ui.mechanismCombo
        self.mechanismCombo.clear()
        self.mechanismCombo.addItems(["Уровни истинности", "Композиция Max-Min", "Композиция Max-Product"])

        # Тип импликации
        self.implCombo = self.ui.implCombo
        self.implCombo.clear()
        self.implCombo.addItems(["Мамдани", "Ларсен"])

        # Тип агрегации
        self.aggCombo = self.ui.aggCombo
        self.aggCombo.clear()
        self.aggCombo.addItems(["MAX", "SUM"])

    def get_implication_type(self):
        return ImplicationType.MAMDANI if self.implCombo.currentIndex() == 0 else ImplicationType.LARSEN

    def get_aggregation_type(self):
        return AggregationType.MAX if self.aggCombo.currentIndex() == 0 else AggregationType.SUM

    def get_composition_type(self):
        index = self.mechanismCombo.currentIndex()
        if index == 1:
            return CompositionType.MAX_MIN
        elif index == 2:
            return CompositionType.MAX_PROD
        return None

    def calculate(self):
        try:
            input_val = self.input1_spin.value()
            inputs = {"количество_предметов": input_val}

            rules_filtered = [
                r for r in self.rules
                if "количество_предметов" in r.conditions and r.result_var == "готовность_к_бою"
            ]
            if not rules_filtered:
                QMessageBox.warning(self, "Предупреждение", "Не найдено правил для данной системы")
                return

            engine = FuzzyInferenceEngine(rules_filtered, self.variables)
            impl_type = self.get_implication_type()
            agg_type = self.get_aggregation_type()
            comp_type = self.get_composition_type()

            truth_levels = engine.get_rule_truth_levels(inputs, "готовность_к_бою")

            self.ui.resultText.clear()
            self.ui.resultText.append(f"Входное значение: {int(input_val)}")

            if comp_type is None:
                # Механизм: уровни истинности
                membership, x_range = engine.inference_truth_level(
                    inputs, "готовность_к_бою", impl_type=impl_type, agg_type=agg_type
                )
                output = engine.defuzzify_centroid(membership, x_range)
            else:
                # Механизм: композиция
                membership, x_range = engine.inference_composition(
                    inputs, "готовность_к_бою",
                    impl_type=impl_type,
                    comp_type=comp_type,
                    agg_type=agg_type
                )
                output = engine.defuzzify_centroid(membership, x_range)

            self.ui.resultText.append(f"Выходное значение: {output:.2f}")
            self.append_truth_levels(truth_levels)
            self.plot_widget.plot_firing_levels(truth_levels)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка вычисления: {e}")
            import traceback
            traceback.print_exc()

    def append_truth_levels(self, truth_levels):
        self.ui.resultText.append("\nУровни истинности правил:")
        if not truth_levels:
            self.ui.resultText.append("  Правила не активированы.")
        else:
            for idx, (rule, alpha) in enumerate(truth_levels, 1):
                self.ui.resultText.append(f"  {idx}. {rule}")
                self.ui.resultText.append(f"      α = {alpha:.3f}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
