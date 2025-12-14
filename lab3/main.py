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
from mpl_toolkits.mplot3d import Axes3D
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

    def plot_membership_functions(self, input_vars, output_var, input_values=None, 
                                  output_membership=None, output_range=None, title="Функции принадлежности"):
        """Отображение входных и выходных функций принадлежности"""
        self.figure.clear()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        if len(input_vars) == 1:
            # Старый вариант для одного входа
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212)
            input_var = input_vars[0]
            x_range = np.linspace(input_var.min_val, input_var.max_val, 300)
            for idx, (term_name, term) in enumerate(input_var.terms.items()):
                y_vals = np.array([term.membership(x) for x in x_range])
                ax1.plot(x_range, y_vals, linewidth=2, label=term_name, color=colors[idx % len(colors)])
                ax1.fill_between(x_range, y_vals, alpha=0.2, color=colors[idx % len(colors)])
            if input_values and input_var.name in input_values:
                ax1.axvline(x=input_values[input_var.name], color='black', linestyle='--', linewidth=2,
                            label=f'Вход = {input_values[input_var.name]}')
            ax1.set_title(f'Входная переменная: {input_var.name}')
            ax1.set_ylim(0, 1.05)
            ax1.set_ylabel('Степень принадлежности')
            ax1.legend(loc='upper right', fontsize=9)
            ax1.grid(True, alpha=0.3)

            # Выход
            if output_range is not None and output_membership is not None:
                ax2.fill_between(output_range, output_membership, alpha=0.3, color='#45B7D1', label='Выходная функция принадлежности')
                ax2.plot(output_range, output_membership, linewidth=2, color='#45B7D1')
            for idx, (term_name, term) in enumerate(output_var.terms.items()):
                x_range_out = np.linspace(output_var.min_val, output_var.max_val, 300)
                y_vals = np.array([term.membership(x) for x in x_range_out])
                ax2.plot(x_range_out, y_vals, linewidth=1.5, label=term_name,
                        color=colors[idx % len(colors)], linestyle=':', alpha=0.7)
            ax2.set_title(f'Выходная переменная: {output_var.name}')
            ax2.set_ylim(0, 1.05)
            ax2.set_xlabel('Значение')
            ax2.set_ylabel('Степень принадлежности')
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3)

        else:
            # Новый вариант для 2+ входов
            axes = self.figure.subplots(2, 2).flatten()
            # Верх: первые два входа
            for i, input_var in enumerate(input_vars[:2]):
                ax = axes[i]
                x_range = np.linspace(input_var.min_val, input_var.max_val, 300)
                for idx, (term_name, term) in enumerate(input_var.terms.items()):
                    y_vals = np.array([term.membership(x) for x in x_range])
                    ax.plot(x_range, y_vals, linewidth=2, label=term_name, color=colors[idx % len(colors)])
                    ax.fill_between(x_range, y_vals, alpha=0.2, color=colors[idx % len(colors)])
                if input_values and input_var.name in input_values:
                    ax.axvline(x=input_values[input_var.name], color='black', linestyle='--', linewidth=2,
                            label=f'Вход = {input_values[input_var.name]}')
                ax.set_title(f'Входная переменная: {input_var.name}')
                ax.set_ylim(0, 1.05)
                ax.set_ylabel('Степень принадлежности')
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)

            # Нижний левый график — третий вход (если есть)
            ax3 = axes[2]
            if len(input_vars) > 2:
                input_var = input_vars[2]
                x_range = np.linspace(input_var.min_val, input_var.max_val, 300)
                for idx, (term_name, term) in enumerate(input_var.terms.items()):
                    y_vals = np.array([term.membership(x) for x in x_range])
                    ax3.plot(x_range, y_vals, linewidth=2, label=term_name, color=colors[idx % len(colors)])
                    ax3.fill_between(x_range, y_vals, alpha=0.2, color=colors[idx % len(colors)])
                if input_values and input_var.name in input_values:
                    ax3.axvline(x=input_values[input_var.name], color='black', linestyle='--', linewidth=2,
                                label=f'Вход = {input_values[input_var.name]}')
                ax3.set_title(f'Входная переменная: {input_var.name}')
                ax3.set_ylim(0, 1.05)
                ax3.set_ylabel('Степень принадлежности')
                ax3.legend(loc='upper right', fontsize=9)
                ax3.grid(True, alpha=0.3)
            else:
                ax3.axis('off')

            # Нижний правый график — выходная переменная
            ax4 = axes[3]
            if output_range is not None and output_membership is not None:
                ax4.fill_between(output_range, output_membership, alpha=0.3, color='#45B7D1', label='Выходная функция принадлежности')
                ax4.plot(output_range, output_membership, linewidth=2, color='#45B7D1')
            for idx, (term_name, term) in enumerate(output_var.terms.items()):
                x_range_out = np.linspace(output_var.min_val, output_var.max_val, 300)
                y_vals = np.array([term.membership(x) for x in x_range_out])
                ax4.plot(x_range_out, y_vals, linewidth=1.5, label=term_name,
                        color=colors[idx % len(colors)], linestyle=':', alpha=0.7)
            ax4.set_title(f'Выходная переменная: {output_var.name}')
            ax4.set_ylim(0, 1.05)
            ax4.set_xlabel('Значение')
            ax4.set_ylabel('Степень принадлежности')
            ax4.legend(loc='upper right', fontsize=9)
            ax4.grid(True, alpha=0.3)

        self.figure.tight_layout()
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
        self.model_file = MODEL_FILE  # Текущий файл конфигурации
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
    
    def _update_rules_spinbox_max(self):
        """Обновляет максимум для rulesSpinBox на основе доступных правил из первой модели"""
        try:
            # Загружаем первую модель для определения количества правил
            variables, rules, input_vars, output_var = self.parser.parse_file(
                self.model_file, model_index=0
            )
            
            # Фильтруем правила для одной входной переменной
            input_var_name = "количество_предметов"
            rules_filtered = [
                r for r in rules
                if input_var_name in r.conditions and r.result_var == output_var
            ]
            
            # Устанавливаем максимум
            max_rules = max(3, len(rules_filtered))  # Минимум 3, но не меньше доступных правил
            if hasattr(self, 'ui') and hasattr(self.ui, 'rulesSpinBox'):
                self.ui.rulesSpinBox.setMaximum(max_rules)
                # Если текущее значение больше максимума, устанавливаем максимум
                if self.ui.rulesSpinBox.value() > max_rules:
                    self.ui.rulesSpinBox.setValue(max_rules)
        except Exception as e:
            # В случае ошибки оставляем значение по умолчанию
            if hasattr(self, 'ui') and hasattr(self.ui, 'rulesSpinBox'):
                self.ui.rulesSpinBox.setMaximum(20)  # Значение по умолчанию

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
        self.ui.buildModelBtn.clicked.connect(self.compute_models_part2)
        
        # Устанавливаем максимум для количества правил на основе загруженных данных
        self._update_rules_spinbox_max()

    def on_system_changed(self, index):
        """Обработчик изменения типа системы"""
        if hasattr(self, 'model_file') and self.model_file:
            try:
                # Загружаем модель из файла с новым индексом
                variables, rules, input_vars, output_var = self.parser.parse_file(
                    self.model_file, 
                    model_index=index
                )
                
                # Обновляем все переменные состояния
                self.variables = variables
                self.rules = rules
                self.input_variables = input_vars
                self.output_variable = output_var
                self.current_model_index = index
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель {index}: {e}")
        
        # Пересоздаём виджеты для новой модели
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
            
            # Сохраняем индекс текущей модели и путь к файлу
            self.current_model_index = model_index
            self.model_file = file_path  # ← ДОБАВЛЕНО: сохраняем путь для будущих переключений
            
            # Обновляем combo box с правильным индексом модели
            self.ui.systemCombo.blockSignals(True)
            self.ui.systemCombo.setCurrentIndex(model_index)
            self.ui.systemCombo.blockSignals(False)
            
            # Пересоздаём входные виджеты под новую модель
            self.create_input_widgets()
            
            # Обновляем максимум для количества правил
            self._update_rules_spinbox_max()
            
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

            layout.addWidget(QLabel("Количество врагов рядом (1-5):"))
            self.input2_spin = QSpinBox()
            self.input2_spin.setRange(1, 5)
            self.input2_spin.setValue(1)
            layout.addWidget(self.input2_spin)

            layout.addWidget(QLabel("Количество союзников рядом (1-5):"))
            self.input3_spin = QSpinBox()
            self.input3_spin.setRange(1, 5)
            self.input3_spin.setValue(1)
            layout.addWidget(self.input3_spin)

        # Механизм логического вывода
        self.mechanismCombo = self.ui.mechanismCombo
        self.mechanismCombo.clear()
        
        system_index = self.ui.systemCombo.currentIndex()
        if system_index == 0:
            # Для системы 1 вход/1 выход доступны все механизмы
            self.mechanismCombo.addItems([
                "Max-Min композиция",
                "Max-Product композиция",
                "Уровни истинности предпосылок"
            ])
            self.mechanismCombo.setEnabled(True)
        else:
            # Для систем с несколькими входами только уровни истинности предпосылок
            self.mechanismCombo.addItems([
                "Уровни истинности предпосылок"
            ])
            self.mechanismCombo.setEnabled(False)


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
            input_vars_objects = [self.variables[name] for name in inputs.keys() if name in self.variables]
            output_variable = self.variables.get(output_var)
            if output_variable and input_vars_objects:
                title = "Функции принадлежности"
                self.plot_widget_part1.plot_membership_functions(
                    input_vars=input_vars_objects,
                    output_var=output_variable,
                    input_values=inputs,
                    output_membership=membership,
                    output_range=x_range,
                    title=title
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
        # Для систем с несколькими входами всегда используем уровни истинности предпосылок
        num_inputs = len(inputs)
        
        # Для многовходовых систем механизм всегда "уровни истинности" (индекс 0 в списке)
        if num_inputs > 1:
            mechanism_index = 0  # уровни истинности
        
        if num_inputs == 1 and mechanism_index == 0:
            # Механизм: Max-Min композиция (только для 1 входа)
            membership, x_range = engine.inference_composition(
                inputs, output_var,
                comp_type=CompositionType.MAX_MIN,
                impl_type=impl_type,
                agg_type=agg_type
            )
            return engine.defuzzify_centroid(membership, x_range)
        elif num_inputs == 1 and mechanism_index == 1:
            # Механизм: Max-Product композиция (только для 1 входа)
            membership, x_range = engine.inference_composition(
                inputs, output_var,
                comp_type=CompositionType.MAX_PROD,
                impl_type=impl_type,
                agg_type=agg_type
            )
            return engine.defuzzify_centroid(membership, x_range)
        else:
            # Механизм: уровни истинности предпосылок (для многовходовых систем или если выбран этот механизм)
            membership, x_range = engine.inference_truth_level(
                inputs, output_var, impl_type=impl_type, agg_type=agg_type
            )
            return engine.defuzzify_centroid(membership, x_range)

    def _compute_output_with_membership(self, engine: FuzzyInferenceEngine, inputs: dict, 
                                       output_var: str, mechanism_index: int, 
                                       impl_type, agg_type) -> tuple:
        """Вычисляет выходное значение с возвращением функции принадлежности"""
        # Для систем с несколькими входами всегда используем уровни истинности предпосылок
        num_inputs = len(inputs)
        
        # Для многовходовых систем механизм всегда "уровни истинности" (индекс 0 в списке)
        if num_inputs > 1:
            mechanism_index = 0  # уровни истинности
        
        if num_inputs == 1 and mechanism_index == 0:
            # Механизм: Max-Min композиция (только для 1 входа)
            membership, x_range = engine.inference_composition(
                inputs, output_var,
                comp_type=CompositionType.MAX_MIN,
                impl_type=impl_type,
                agg_type=agg_type
            )
            output = engine.defuzzify_centroid(membership, x_range)
            return output, membership, x_range
        elif num_inputs == 1 and mechanism_index == 1:
            # Механизм: Max-Product композиция (только для 1 входа)
            membership, x_range = engine.inference_composition(
                inputs, output_var,
                comp_type=CompositionType.MAX_PROD,
                impl_type=impl_type,
                agg_type=agg_type
            )
            output = engine.defuzzify_centroid(membership, x_range)
            return output, membership, x_range
        else:
            # Механизм: уровни истинности предпосылок (для многовходовых систем или если выбран этот механизм)
            membership, x_range = engine.inference_truth_level(
                inputs, output_var, impl_type=impl_type, agg_type=agg_type
            )
            output = engine.defuzzify_centroid(membership, x_range)
            return output, membership, x_range

    def compare_implications(self):
        """Сравнение типов импликаций (Мамдани и Ларсен) с учётом выбранного механизма"""
        try:
            system_index = self.ui.systemCombo.currentIndex()
            
            # Получаем входные данные для текущей системы
            inputs, input_vars, output_var = self._get_system_data(system_index)
            if inputs is None:
                return
            
            mechanism_index = self.mechanismCombo.currentIndex()
            agg_type = self.get_aggregation_type()

            # Фильтруем правила для данной системы
            rules_filtered = self._filter_rules_for_system(inputs, output_var)
            if not rules_filtered:
                QMessageBox.warning(self, "Предупреждение", "Не найдено правил для данной системы")
                return

            # Названия механизмов и типов агрегации
            num_inputs = len(inputs)
            
            # Для многовходовых систем механизм всегда "уровни истинности"
            if num_inputs > 1:
                actual_mechanism_index = 0
            else:
                actual_mechanism_index = mechanism_index
            
            if num_inputs == 1:
                mechanism_names = ["Max-Min композиция", "Max-Product композиция", "Уровни истинности предпосылок"]
                mechanism_name = mechanism_names[actual_mechanism_index]
            else:
                mechanism_name = "Уровни истинности предпосылок"
            
            agg_names = ["MAX", "SUM", "PROBOR"]
            agg_name = agg_names[self.aggCombo.currentIndex()]

            engine = FuzzyInferenceEngine(rules_filtered, self.variables)

            self.ui.resultText.clear()
            self._append_input_values(inputs, system_index)
            self.ui.resultText.append(f"Механизм вывода: {mechanism_name}")
            self.ui.resultText.append(f"Тип агрегации: {agg_name}")
            self.ui.resultText.append("")

            # Вычисляем выходы для обоих типов импликаций
            for impl_name, impl_type in [("Мамдани", ImplicationType.MAMDANI), ("Ларсен", ImplicationType.LARSEN)]:
                if num_inputs == 1 and actual_mechanism_index == 0:
                    # Max-Min композиция (только для 1 входа)
                    membership, x_range = engine.inference_composition(
                        inputs, output_var,
                        comp_type=CompositionType.MAX_MIN,
                        impl_type=impl_type,
                        agg_type=agg_type
                    )
                elif num_inputs == 1 and actual_mechanism_index == 1:
                    # Max-Product композиция (только для 1 входа)
                    membership, x_range = engine.inference_composition(
                        inputs, output_var,
                        comp_type=CompositionType.MAX_PROD,
                        impl_type=impl_type,
                        agg_type=agg_type
                    )
                else:
                    # Уровни истинности предпосылок (для многовходовых систем или если выбран этот механизм)
                    membership, x_range = engine.inference_truth_level(
                        inputs, output_var,
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

    def _parse_function(self, func_str: str):
        """Безопасный парсинг функции из строки"""
        import math
        # Разрешаем только безопасные функции
        safe_dict = {
            "x": None,  # Будет заменено на значение
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "exp": math.exp,
            "log": math.log,
            "sqrt": math.sqrt,
            "abs": abs,
            "pi": math.pi,
            "e": math.e,
            "__builtins__": {},
        }
        return lambda x: eval(func_str, {"__builtins__": {}}, {**safe_dict, "x": x})
    
    def compute_models_part2(self):
        """Вычисление моделей Мамдани и Такаги-Сугено для части 2 с правильным алгоритмом"""
        try:
            # Получаем параметры
            func_str = self.ui.functionLineEdit.text().strip()
            if not func_str:
                QMessageBox.warning(self, "Ошибка", "Введите функцию")
                return
            
            a = self.ui.aSpinBox.value()
            b = self.ui.bSpinBox.value()
            if a >= b:
                QMessageBox.warning(self, "Ошибка", "a должно быть меньше b")
                return
            
            num_rules = self.ui.rulesSpinBox.value()
            resolution = 100  # Фиксированное разрешение для вычислений
            
            # Парсим функцию
            try:
                func = self._parse_function(func_str)
                # Проверяем функцию на тестовом значении
                test_val = (a + b) / 2
                func(test_val)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Неверный формат функции: {e}")
                return
            
            # Загружаем модель (используем первую модель для правил)
            try:
                variables, rules, input_vars, output_var = self.parser.parse_file(
                    self.model_file, model_index=0
                )
            except:
                QMessageBox.critical(self, "Ошибка", "Не удалось загрузить модель")
                return
            
            # Фильтруем правила для одной входной переменной
            input_var_name = "количество_предметов"  # Используем переменную из первой модели
            rules_filtered = [
                r for r in rules
                if input_var_name in r.conditions and r.result_var == output_var
            ]
            
            if len(rules_filtered) < 3:
                QMessageBox.warning(self, "Ошибка", f"Недостаточно правил (нужно минимум 3, найдено {len(rules_filtered)})")
                return
            
            # Ограничиваем количество правил
            if len(rules_filtered) > num_rules:
                rules_filtered = rules_filtered[:num_rules]
            
            # Создаём движок
            engine = FuzzyInferenceEngine(rules_filtered, variables)
            
            # Вычисляем исходную функцию на интервале
            x_range = np.linspace(a, b, resolution)
            y_original = np.array([func(x) for x in x_range])
            
            # Вычисляем диапазон исходной функции
            y_min, y_max = y_original.min(), y_original.max()
            output_variable = variables[output_var]
            output_range_model = (output_variable.min_val, output_variable.max_val)
            
            # Масштабируем входные значения к диапазону входной переменной модели
            input_var = variables[input_var_name]
            input_range_model = (input_var.min_val, input_var.max_val)
            
            # ===== МОДЕЛЬ МАМДАНИ =====
            # Алгоритм:
            # 1. Вычисление уровней истинности предпосылок для каждого правила
            # 2. Вычисление выходов для каждого правила на основе импликации (min для Мамдани)
            # 3. Агрегация индивидуальных выходов (max) и дефазификация методом центра тяжести
            y_mamdani = []
            for x in x_range:
                # Масштабируем x к диапазону входной переменной модели
                x_scaled = input_range_model[0] + (x - a) / (b - a) * (input_range_model[1] - input_range_model[0])
                inputs = {input_var_name: x_scaled}
                
                # Используем inference_truth_level с импликацией Мамдани и агрегацией MAX
                membership, x_out_range = engine.inference_truth_level(
                    inputs, output_var, 
                    impl_type=ImplicationType.MAMDANI, 
                    agg_type=AggregationType.MAX
                )
                # Дефазификация методом центра тяжести
                output = engine.defuzzify_centroid(membership, x_out_range)
                # Масштабируем выход модели к диапазону исходной функции
                output_scaled = y_min + (output - output_range_model[0]) / (output_range_model[1] - output_range_model[0]) * (y_max - y_min)
                y_mamdani.append(output_scaled)
            y_mamdani = np.array(y_mamdani)
            
            # ===== МОДЕЛЬ ТАКАГИ-СУГЕНО =====
            # Алгоритм:
            # 1. Вычисление уровней истинности предпосылок для каждого правила
            # 2. Вычисление выходов для каждого правила на основе импликации (prod для Ларсена)
            # 3. Агрегация индивидуальных выходов методом центра тяжести (взвешенная сумма)
            
            # Создаём функции для правил
            # Для Такаги-Сугено функции правил должны возвращать значения в масштабе выходной переменной модели
            rule_functions = {}
            for rule_idx, rule in enumerate(rules_filtered):
                # Находим центр терма входной переменной
                input_term_name = rule.conditions[input_var_name]
                input_term = input_var.terms[input_term_name]
                
                # Находим центр терма (максимум функции принадлежности)
                x_input_range = np.linspace(input_var.min_val, input_var.max_val, 100)
                memberships = np.array([input_term.membership(x) for x in x_input_range])
                center_idx = np.argmax(memberships)
                x_center_model = x_input_range[center_idx]
                
                # Масштабируем центр обратно к исходному диапазону [a, b]
                x_center = a + (x_center_model - input_range_model[0]) / (input_range_model[1] - input_range_model[0]) * (b - a)
                
                # Вычисляем значение исходной функции в центре
                y_center_original = func(x_center)
                
                # Масштабируем значение исходной функции к диапазону выходной переменной модели
                y_center_model = output_range_model[0] + (y_center_original - y_min) / (y_max - y_min) * (output_range_model[1] - output_range_model[0])
                
                # Создаём функцию правила, которая возвращает значение в масштабе выходной переменной модели
                def make_rule_func(rule_idx_val, yc_model, input_min, input_max, output_min, output_max, a_val, b_val, y_min_val, y_max_val, func_ref):
                    def rule_func(x_model):
                        # Масштабируем x_model обратно к [a, b]
                        x_orig = a_val + (x_model - input_min) / (input_max - input_min) * (b_val - a_val)
                        # Вычисляем значение исходной функции
                        y_orig = func_ref(x_orig)
                        # Масштабируем к диапазону выходной переменной модели
                        y_model = output_min + (y_orig - y_min_val) / (y_max_val - y_min_val) * (output_max - output_min)
                        return y_model
                    return rule_func
                
                rule_functions[rule_idx] = make_rule_func(
                    rule_idx, y_center_model, 
                    input_range_model[0], input_range_model[1],
                    output_range_model[0], output_range_model[1],
                    a, b, y_min, y_max, func
                )
            
            y_sugeno = []
            for x in x_range:
                # Масштабируем x к диапазону входной переменной модели
                x_scaled = input_range_model[0] + (x - a) / (b - a) * (input_range_model[1] - input_range_model[0])
                inputs = {input_var_name: x_scaled}
                
                # Для Такаги-Сугено используем prod для импликации (как Ларсен)
                # inference_takagi_sugeno уже реализует взвешенную сумму
                output = engine.inference_takagi_sugeno(
                    inputs, output_var, rule_functions, agg_type=AggregationType.MAX
                )
                # Масштабируем выход модели к диапазону исходной функции
                output_scaled = y_min + (output - output_range_model[0]) / (output_range_model[1] - output_range_model[0]) * (y_max - y_min)
                y_sugeno.append(output_scaled)
            y_sugeno = np.array(y_sugeno)
            
            # Сохраняем результаты
            self.part2_data = {
                'x_range': x_range,
                'y_original': y_original,
                'y_mamdani': y_mamdani,
                'y_sugeno': y_sugeno,
                'func': func,
                'a': a,
                'b': b,
                'rules_count': len(rules_filtered)
            }
            
            # Выводим результаты и строим графики
            self._display_results_part2(func_str, a, b, resolution)
            self._plot_comparison_part2()
            self._plot_surfaces_part2()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка вычисления моделей: {e}")
            import traceback
            traceback.print_exc()
    
    def _display_results_part2(self, func_str, a, b, resolution):
        """Отображение результатов вычислений"""
        data = self.part2_data
        y_original = data['y_original']
        y_mamdani = data['y_mamdani']
        y_sugeno = data['y_sugeno']
        
        # Вычисляем метрики
        mse_mamdani = np.mean((y_original - y_mamdani) ** 2)
        mse_sugeno = np.mean((y_original - y_sugeno) ** 2)
        mae_mamdani = np.mean(np.abs(y_original - y_mamdani))
        mae_sugeno = np.mean(np.abs(y_original - y_sugeno))
        
        self.ui.resultTextPart2.clear()
        self.ui.resultTextPart2.append(f"Функция: y = {func_str}")
        self.ui.resultTextPart2.append(f"Интервал: [{a:.2f}, {b:.2f}]")
        self.ui.resultTextPart2.append(f"Количество правил: {data['rules_count']}")
        self.ui.resultTextPart2.append(f"Разрешение: {resolution}")
        self.ui.resultTextPart2.append("\n" + "="*50)
        self.ui.resultTextPart2.append("РЕЗУЛЬТАТЫ ВЫЧИСЛЕНИЙ:\n")
        self.ui.resultTextPart2.append(f"Модель Мамдани:")
        self.ui.resultTextPart2.append(f"  MSE (среднеквадратичная ошибка): {mse_mamdani:.6f}")
        self.ui.resultTextPart2.append(f"  MAE (средняя абсолютная ошибка): {mae_mamdani:.6f}")
        self.ui.resultTextPart2.append(f"\nМодель Такаги-Сугено:")
        self.ui.resultTextPart2.append(f"  MSE (среднеквадратичная ошибка): {mse_sugeno:.6f}")
        self.ui.resultTextPart2.append(f"  MAE (средняя абсолютная ошибка): {mae_sugeno:.6f}")
    
    def _plot_comparison_part2(self):
        """Построение графика сравнения моделей с исходной функцией"""
        if not hasattr(self, 'part2_data'):
            return
        
        data = self.part2_data
        x = data['x_range']
        y_original = data['y_original']
        y_mamdani = data['y_mamdani']
        y_sugeno = data['y_sugeno']
        
        # Создаём 3 подграфика: сверху сравнение (2 столбца), снизу 2 поверхности
        self.plot_widget_part2.figure.clear()
        
        # Верхний график: сравнение (занимает 2 столбца)
        ax1 = self.plot_widget_part2.figure.add_subplot(211)
        ax1.plot(x, y_original, 'b-', linewidth=2, label='Исходная функция', alpha=0.8)
        ax1.plot(x, y_mamdani, 'r--', linewidth=2, label='Модель Мамдани', alpha=0.8)
        ax1.plot(x, y_sugeno, 'g:', linewidth=2, label='Модель Такаги-Сугено', alpha=0.8)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Сравнение моделей с исходной функцией')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        self.plot_widget_part2.figure.tight_layout()
        self.plot_widget_part2.canvas.draw()
    
    def _plot_surfaces_part2(self):
        """Построение поверхностей отображения для обеих моделей"""
        if not hasattr(self, 'part2_data'):
            return
        
        data = self.part2_data
        x = data['x_range']
        y_mamdani = data['y_mamdani']
        y_sugeno = data['y_sugeno']
        num_rules = data['rules_count']
        
        # Создаём поверхности отображения: x, количество правил, y
        # X - входное значение x
        # Y - количество правил (от 1 до num_rules)
        # Z - выходное значение y модели
        
        # Создаём сетку для поверхности
        num_points = len(x)
        rules_range = np.arange(1, num_rules + 1)
        
        # Создаём 2D сетку
        X_mamdani, Y_mamdani = np.meshgrid(x, rules_range)
        X_sugeno, Y_sugeno = np.meshgrid(x, rules_range)
        
        # Z координата - выходное значение y модели (одинаковое для всех правил, так как мы используем все правила)
        Z_mamdani = np.tile(y_mamdani, (num_rules, 1))
        Z_sugeno = np.tile(y_sugeno, (num_rules, 1))
        
        # Нижний левый график: поверхность Мамдани
        ax2 = self.plot_widget_part2.figure.add_subplot(223, projection='3d')
        ax2.plot_surface(X_mamdani, Y_mamdani, Z_mamdani, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)
        ax2.plot(x, np.ones_like(x), y_mamdani, 'r-', linewidth=2, label='Модель Мамдани')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Количество правил')
        ax2.set_zlabel('y')
        ax2.set_title('Поверхность отображения: Мамдани')
        
        # Нижний правый график: поверхность Такаги-Сугено
        ax3 = self.plot_widget_part2.figure.add_subplot(224, projection='3d')
        ax3.plot_surface(X_sugeno, Y_sugeno, Z_sugeno, cmap='plasma', alpha=0.7, linewidth=0, antialiased=True)
        ax3.plot(x, np.ones_like(x), y_sugeno, 'g-', linewidth=2, label='Модель Такаги-Сугено')
        ax3.set_xlabel('x')
        ax3.set_ylabel('Количество правил')
        ax3.set_zlabel('y')
        ax3.set_title('Поверхность отображения: Такаги-Сугено')
        
        self.plot_widget_part2.figure.tight_layout()
        self.plot_widget_part2.canvas.draw()



def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()