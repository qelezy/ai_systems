"""
GUI приложение для системы нечёткого логического вывода с выбором механизма
"""
import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QSpinBox, QMessageBox, QDoubleSpinBox, QFileDialog, QInputDialog
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
        self.figure = Figure(figsize=(6, 4), dpi=80)
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_firing_levels(self, truth_levels, title="Уровни истинности правил"):
        """Отображение уровней истинности правил"""
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

    def plot_membership_functions(self, input_var, output_var, input_value=None, 
                                  output_membership=None, output_range=None, title="Функции принадлежности"):
        """Отображение входных и выходных функций принадлежности"""
        self.figure.clear()
        
        # Создаём два подграфика: один для входа (сверху), один для выхода (снизу)
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)
        
        # ===== Входная переменная (СВЕРХУ) =====
        x_range = np.linspace(input_var.min_val, input_var.max_val, 300)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        for idx, (term_name, term) in enumerate(input_var.terms.items()):
            y_vals = np.array([term.membership(x) for x in x_range])
            ax1.plot(x_range, y_vals, linewidth=2, label=term_name, color=colors[idx % len(colors)])
            ax1.fill_between(x_range, y_vals, alpha=0.2, color=colors[idx % len(colors)])
        
        # Отмечаем входное значение вертикальной линией
        if input_value is not None:
            ax1.axvline(x=input_value, color='black', linestyle='--', linewidth=2, label=f'Вход = {input_value}')
        
        ax1.set_xlabel('Значение')
        ax1.set_ylabel('Степень принадлежности')
        ax1.set_title(f'Входная переменная: {input_var.name}')
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # ===== Выходная переменная (СНИЗУ) =====
        if output_range is not None and output_membership is not None:
            # Отображаем результат вывода
            ax2.fill_between(output_range, output_membership, alpha=0.3, color='#45B7D1', label='Выходная функция принадлежности')
            ax2.plot(output_range, output_membership, linewidth=2, color='#45B7D1')
        
        # Отображаем все термы выходной переменной
        for idx, (term_name, term) in enumerate(output_var.terms.items()):
            x_range_out = np.linspace(output_var.min_val, output_var.max_val, 300)
            y_vals = np.array([term.membership(x) for x in x_range_out])
            ax2.plot(x_range_out, y_vals, linewidth=1.5, label=term_name, 
                    color=colors[idx % len(colors)], linestyle=':', alpha=0.7)
        
        ax2.set_xlabel('Значение')
        ax2.set_ylabel('Степень принадлежности')
        ax2.set_title(f'Выходная переменная: {output_var.name}')
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_comparison(self, x_original, y_original, x_mamdani, y_mamdani, 
                       x_sugeno, y_sugeno, title="Сравнение моделей"):
        """Отображение сравнения исходной функции с моделями Мамдани и Такаги-Сугено"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x_original, y_original, 'b-', linewidth=2, label='Исходная функция', alpha=0.8)
        ax.plot(x_mamdani, y_mamdani, 'r--', linewidth=2, label='Модель Мамдани', alpha=0.8)
        ax.plot(x_sugeno, y_sugeno, 'g:', linewidth=2, label='Модель Такаги-Сугено', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def plot_surface(self, x, y, z, title="Поверхность отображения", model_name=""):
        """Отображение 3D поверхности"""
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        if len(x.shape) == 1:
            # Если данные одномерные, создаем сетку
            X, Y = np.meshgrid(x, y)
            Z = np.tile(z, (len(y), 1))
        else:
            X, Y, Z = x, y, z
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f"{title} - {model_name}")
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
        self.current_model_index = 0  # Индекс текущей загруженной модели
        self.load_default_data()

        self.load_ui()
        self.setup_part1()
        self.setup_part2()

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

    def setup_part1(self):
        """Настройка интерфейса для части 1"""
        # График для части 1
        self.plot_widget_part1 = FuzzyPlotWidget()
        plot_layout = QVBoxLayout(self.ui.plotWidgetContainer)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.plot_widget_part1)

        # Подключение сигналов
        self.ui.calculateBtn.clicked.connect(self.calculate_part1)
        self.ui.systemCombo.currentIndexChanged.connect(self.on_system_changed)
        self.ui.compareImplBtn.clicked.connect(self.compare_implications)
        self.ui.loadBtn.clicked.connect(self.load_data_from_file)

        # Инициализация виджетов
        self.create_input_widgets()

    def setup_part2(self):
        """Настройка интерфейса для части 2"""
        # График для части 2
        self.plot_widget_part2 = FuzzyPlotWidget()
        plot_layout = QVBoxLayout(self.ui.plotWidgetContainerPart2)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.plot_widget_part2)

        # Подключение сигналов
        self.ui.buildModelBtn.clicked.connect(self.build_models_part2)
        self.ui.compareModelsBtn.clicked.connect(self.compare_models_part2)
        self.ui.plotSurfaceBtn.clicked.connect(self.plot_surfaces_part2)

    def on_system_changed(self, index):
        """Обработчик изменения типа системы"""
        self.create_input_widgets()

    def load_data_from_file(self):
        """Загрузка данных из JSON файла"""
        try:
            # Открываем диалог выбора файла
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Выберите JSON файл с конфигурацией",
                "",
                "JSON файлы (*.json);;Все файлы (*)"
            )
            
            if not file_path:
                return
            
            # Получаем список доступных моделей
            available_models = self.parser.get_available_models(file_path)
            
            if not available_models:
                QMessageBox.warning(self, "Ошибка", "В файле не найдены модели")
                return
            
            # Определяем индекс модели для загрузки
            if len(available_models) == 1:
                # Если только одна модель - загружаем её
                model_index = 0
            else:
                # Если несколько моделей - показываем диалог выбора
                model_names = [name for _, name, _ in available_models]
                selected_name, ok = QInputDialog.getItem(
                    self,
                    "Выбор модели",
                    "Выберите модель для загрузки:",
                    model_names,
                    0,
                    False
                )
                
                if not ok:
                    return
                
                model_index = model_names.index(selected_name)
            
            # Загружаем выбранную модель
            self.variables, self.rules, self.input_variables, self.output_variable = \
                self.parser.parse_file(file_path, model_index=model_index)
            
            # Сохраняем индекс текущей модели
            self.current_model_index = model_index
            
            # Пересоздаём входные виджеты под новую модель
            self.create_input_widgets()
            
            # Очищаем результаты
            self.ui.resultText.clear()
            
            # Уведомляем пользователя об успешной загрузке
            QMessageBox.information(
                self,
                "Успешно",
                f"Данные загружены из файла:\n{file_path}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл: {e}")
            import traceback
            traceback.print_exc()

    def create_input_widgets(self):
        """Создание виджетов ввода в зависимости от типа системы"""
        layout = self.ui.inputLayout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        system_index = self.ui.systemCombo.currentIndex()

        if system_index == 0:
            # Система 1 вход / 1 выход
            layout.addWidget(QLabel("Количество предметов (0-6):"))
            self.input1_spin = QSpinBox()
            self.input1_spin.setRange(0, 6)
            self.input1_spin.setValue(0)
            layout.addWidget(self.input1_spin)
        else:
            # Система несколько входов / 1 выход
            layout.addWidget(QLabel("Количество предметов (0-6):"))
            self.input1_spin = QSpinBox()
            self.input1_spin.setRange(0, 6)
            self.input1_spin.setValue(0)
            layout.addWidget(self.input1_spin)

            layout.addWidget(QLabel("Количество врагов рядом (0-5):"))
            self.input2_spin = QSpinBox()
            self.input2_spin.setRange(0, 5)
            self.input2_spin.setValue(0)
            layout.addWidget(self.input2_spin)

            layout.addWidget(QLabel("Количество союзников рядом (0-5):"))
            self.input3_spin = QSpinBox()
            self.input3_spin.setRange(0, 5)
            self.input3_spin.setValue(0)
            layout.addWidget(self.input3_spin)

        # Механизм логического вывода
        self.mechanismCombo = self.ui.mechanismCombo
        self.mechanismCombo.clear()
        self.mechanismCombo.addItems([
            "Max-Min композиция",
            "Max-Product композиция",
            "Уровни истинности предпосылок"
        ])


        # Тип импликации
        self.implCombo = self.ui.implCombo
        self.implCombo.clear()
        self.implCombo.addItems(["Мамдани", "Ларсен"])

        # Тип агрегации
        self.aggCombo = self.ui.aggCombo
        self.aggCombo.clear()
        self.aggCombo.addItems(["MAX", "SUM", "PROBOR"])

    def get_implication_type(self):
        return ImplicationType.MAMDANI if self.implCombo.currentIndex() == 0 else ImplicationType.LARSEN

    def get_aggregation_type(self):
        return AggregationType.MAX if self.aggCombo.currentIndex() == 0 else AggregationType.SUM if self.aggCombo.currentIndex() == 1 else AggregationType.PROBOR

    def calculate_part1(self):
        """Вычисление для части 1 - универсальный метод для любого количества входов"""
        try:
            system_index = self.ui.systemCombo.currentIndex()
            impl_type = self.get_implication_type()
            agg_type = self.get_aggregation_type()

            # Получаем входные данные в зависимости от типа системы
            inputs, input_vars, output_var = self._get_system_data(system_index)
            if inputs is None:
                return

            # Фильтруем правила для данной системы
            rules_filtered = self._filter_rules_for_system(inputs, output_var)
            if not rules_filtered:
                QMessageBox.warning(self, "Предупреждение", "Не найдено правил для данной системы")
                return

            # Создаем движок и вычисляем уровни истинности
            engine = FuzzyInferenceEngine(rules_filtered, self.variables)
            truth_levels = engine.get_rule_truth_levels(inputs, output_var)

            # Выводим входные данные
            self.ui.resultText.clear()
            self._append_input_values(inputs, system_index)

            # Вычисляем выходное значение в зависимости от механизма вывода
            mechanism_index = self.mechanismCombo.currentIndex()
            output, membership, x_range = self._compute_output_with_membership(
                engine, inputs, output_var, mechanism_index, impl_type, agg_type
            )

            # Выводим результаты
            self.ui.resultText.append(f"Выходное значение: {output:.4f}")
            self.append_truth_levels(truth_levels)
            
            # Отображаем функции принадлежности
            if system_index == 0:
                input_var = self.variables.get("количество_предметов")
                output_variable = self.variables.get(output_var)
                input_val = list(inputs.values())[0]
                
                if input_var and output_variable:
                    self.plot_widget_part1.plot_membership_functions(
                        input_var, output_variable,
                        input_value=input_val,
                        output_membership=membership,
                        output_range=x_range,
                        title="Функции принадлежности"
                    )

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка вычисления: {e}")
            import traceback
            traceback.print_exc()

    def _get_system_data(self, system_index: int) -> tuple:
        """Получает входные данные, входные переменные и выходную переменную для системы"""
        if system_index == 0:
            # Система 1 вход / 1 выход
            input_val = self.input1_spin.value()
            inputs = {"количество_предметов": input_val}
            input_vars = ["количество_предметов"]
            output_var = "готовность_к_бою"
        else:
            # Система несколько входов / 1 выход
            input1_val = self.input1_spin.value()
            input2_val = self.input2_spin.value()
            input3_val = self.input3_spin.value()
            inputs = {
                "количество_предметов": input1_val,
                "количество_врагов_рядом": input2_val,
                "количество_союзников_рядом": input3_val
            }
            input_vars = list(inputs.keys())
            output_var = "готовность_к_бою"
        
        return inputs, input_vars, output_var

    def _filter_rules_for_system(self, inputs: dict, output_var: str) -> list:
        """Фильтрует правила для данной системы"""
        input_vars = set(inputs.keys())
        return [
            r for r in self.rules
            if all(var in r.conditions for var in input_vars) and r.result_var == output_var
        ]

    def _append_input_values(self, inputs: dict, system_index: int):
        """Выводит входные значения в текстовое поле"""
        if system_index == 0:
            # Один вход
            val = list(inputs.values())[0]
            self.ui.resultText.append(f"Входное значение: {int(val)}")
        else:
            # Несколько входов
            self.ui.resultText.append("Входные значения:")
            for var_name, val in inputs.items():
                # Форматируем имя переменной для вывода
                display_name = var_name.replace("_", " ").title()
                self.ui.resultText.append(f"  {display_name}: {int(val)}")

    def _compute_output(self, engine: FuzzyInferenceEngine, inputs: dict, 
                       output_var: str, mechanism_index: int, 
                       impl_type, agg_type) -> float:
        """Вычисляет выходное значение в зависимости от механизма вывода"""
        if mechanism_index == 0:
            # Механизм: Max-Min композиция
            membership, x_range = engine.inference_composition(
                inputs, output_var,
                comp_type=CompositionType.MAX_MIN,
                impl_type=impl_type,
                agg_type=agg_type
            )
            return engine.defuzzify_centroid(membership, x_range)
        elif mechanism_index == 1:
            # Механизм: Max-Product композиция
            membership, x_range = engine.inference_composition(
                inputs, output_var,
                comp_type=CompositionType.MAX_PROD,
                impl_type=impl_type,
                agg_type=agg_type
            )
            return engine.defuzzify_centroid(membership, x_range)
        else:
            # Механизм: уровни истинности предпосылок (по умолчанию для многовходовых систем)
            membership, x_range = engine.inference_truth_level(
                inputs, output_var, impl_type=impl_type, agg_type=agg_type
            )
            return engine.defuzzify_centroid(membership, x_range)

    def _compute_output_with_membership(self, engine: FuzzyInferenceEngine, inputs: dict, 
                                       output_var: str, mechanism_index: int, 
                                       impl_type, agg_type) -> tuple:
        """Вычисляет выходное значение с возвращением функции принадлежности"""
        if mechanism_index == 0:
            # Механизм: Max-Min композиция
            membership, x_range = engine.inference_composition(
                inputs, output_var,
                comp_type=CompositionType.MAX_MIN,
                impl_type=impl_type,
                agg_type=agg_type
            )
            output = engine.defuzzify_centroid(membership, x_range)
            return output, membership, x_range
        elif mechanism_index == 1:
            # Механизм: Max-Product композиция
            membership, x_range = engine.inference_composition(
                inputs, output_var,
                comp_type=CompositionType.MAX_PROD,
                impl_type=impl_type,
                agg_type=agg_type
            )
            output = engine.defuzzify_centroid(membership, x_range)
            return output, membership, x_range
        else:
            # Механизм: уровни истинности предпосылок
            membership, x_range = engine.inference_truth_level(
                inputs, output_var, impl_type=impl_type, agg_type=agg_type
            )
            output = engine.defuzzify_centroid(membership, x_range)
            return output, membership, x_range

    def compare_implications(self):
        """Сравнение типов импликаций (Мамдани и Ларсен) с учётом выбранного механизма"""
        try:
            system_index = self.ui.systemCombo.currentIndex()
            if system_index != 0:
                QMessageBox.warning(self, "Предупреждение", "Сравнение импликаций доступно только для системы 1 вход/1 выход")
                return

            # Получаем параметры
            input_val = self.input1_spin.value()
            inputs = {"количество_предметов": input_val}
            mechanism_index = self.mechanismCombo.currentIndex()
            agg_type = self.get_aggregation_type()

            # Фильтруем правила
            rules_filtered = [
                r for r in self.rules
                if "количество_предметов" in r.conditions and r.result_var == "готовность_к_бою"
            ]
            if not rules_filtered:
                QMessageBox.warning(self, "Предупреждение", "Не найдено правил для данной системы")
                return

            # Названия механизмов и типов агрегации
            mechanism_names = ["Max-Min композиция", "Max-Product композиция", "Уровни истинности предпосылок"]
            agg_names = ["MAX", "SUM", "PROBOR"]
            mechanism_name = mechanism_names[mechanism_index]
            agg_name = agg_names[self.aggCombo.currentIndex()]

            engine = FuzzyInferenceEngine(rules_filtered, self.variables)

            self.ui.resultText.clear()
            self.ui.resultText.append(f"Входное значение: {int(input_val)}")
            self.ui.resultText.append(f"Механизм вывода: {mechanism_name}")
            self.ui.resultText.append(f"Тип агрегации: {agg_name}")
            self.ui.resultText.append("")

            # Вычисляем выходы для обоих типов импликаций
            for impl_name, impl_type in [("Мамдани", ImplicationType.MAMDANI), ("Ларсен", ImplicationType.LARSEN)]:
                if mechanism_index == 0:
                    # Max-Min композиция
                    membership, x_range = engine.inference_composition(
                        inputs, "готовность_к_бою",
                        comp_type=CompositionType.MAX_MIN,
                        impl_type=impl_type,
                        agg_type=agg_type
                    )
                elif mechanism_index == 1:
                    # Max-Product композиция
                    membership, x_range = engine.inference_composition(
                        inputs, "готовность_к_бою",
                        comp_type=CompositionType.MAX_PROD,
                        impl_type=impl_type,
                        agg_type=agg_type
                    )
                else:
                    # Уровни истинности предпосылок
                    membership, x_range = engine.inference_truth_level(
                        inputs, "готовность_к_бою",
                        impl_type=impl_type,
                        agg_type=agg_type
                    )

                output = engine.defuzzify_centroid(membership, x_range)
                self.ui.resultText.append(f"{impl_name}: {output:.2f}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка сравнения: {e}")
            import traceback
            traceback.print_exc()

    def append_truth_levels(self, truth_levels):
        """Добавление уровней истинности в текстовое поле"""
        self.ui.resultText.append("\nУровни истинности правил:")
        if not truth_levels:
            self.ui.resultText.append("  Правила не активированы.")
        else:
            for idx, (rule, alpha) in enumerate(truth_levels, 1):
                self.ui.resultText.append(f"  {idx}. {rule}")
                self.ui.resultText.append(f"      α = {alpha:.3f}")

    def build_models_part2(self):
        """Построение моделей Мамдани и Такаги-Сугено для части 2"""
        self.ui.resultTextPart2.clear()
        self.ui.resultTextPart2.append("Функционал построения моделей будет реализован в следующем этапе.")
        QMessageBox.information(self, "Информация", "Функционал построения моделей будет реализован позже.")

    def compare_models_part2(self):
        """Сравнение моделей с исходной функцией для части 2"""
        self.ui.resultTextPart2.clear()
        self.ui.resultTextPart2.append("Функционал сравнения моделей будет реализован в следующем этапе.")
        QMessageBox.information(self, "Информация", "Функционал сравнения моделей будет реализован позже.")

    def plot_surfaces_part2(self):
        """Построение поверхностей для части 2"""
        self.ui.resultTextPart2.clear()
        self.ui.resultTextPart2.append("Функционал построения поверхностей будет реализован в следующем этапе.")
        QMessageBox.information(self, "Информация", "Функционал построения поверхностей будет реализован позже.")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()