from fractions import Fraction
from typing import List, Dict, Tuple
import math
import matplotlib.pyplot as plt
import numpy as np

class AHPHouse:
    # Добавляем константу случайной согласованности
    RANDOM_CONSISTENCY = {
        1: 0.0,
        2: 0.0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49
    }
    
    def __init__(self):
        self.criteria = [
            'Размеры дома',
            'Удобство автобусных маршрутов',
            'Окрестности',
            'Когда построен дом',
            'Двор',
            'Современное оборудование',
            'Общее состояние',
            'Финансовые условия'
        ]
        self.alternatives = ['Дом А', 'Дом Б', 'Дом В']
        # Инициализация матрицы парных сравнений критериев
        self.criteria_matrix = self._initialize_matrix(len(self.criteria))
        # Инициализация матриц парных сравнений альтернатив для каждого критерия
        self.alternatives_matrices = {
            criterion: self._initialize_matrix(len(self.alternatives))
            for criterion in self.criteria
        }

    def _initialize_matrix(self, size: int):
        """Инициализация матрицы заданного размера"""
        return [[Fraction(1) for _ in range(size)] for _ in range(size)]

    def set_criteria_comparison(self, i: int, j: int, value: str):
        """Усановка значения сравнения критериев"""
        frac_value = Fraction(value)
        self.criteria_matrix[i][j] = frac_value
        self.criteria_matrix[j][i] = Fraction(1) / frac_value

    def set_alternative_comparison(self, criterion: str, i: int, j: int, value: str):
        """Установка значения сравнения альтернатив для заданного критерия"""
        frac_value = Fraction(value)
        self.alternatives_matrices[criterion][i][j] = frac_value
        self.alternatives_matrices[criterion][j][i] = Fraction(1) / frac_value

    def _format_matrix_as_table(self, headers: List[str], matrix: List[List[Fraction]]) -> List[str]:
        """Форматирование матрицы в markdown-таблицу"""
        md = []
        
        # Заголовок таблицы
        header = "| Варианты | " + " | ".join(headers) + " |"
        separator = "|" + "|".join(["-" * 15 for _ in range(len(headers) + 1)]) + "|"
        
        md.append(header)
        md.append(separator)

        # Строки таблицы
        for i, header in enumerate(headers):
            row = [header]
            for j in range(len(matrix)):
                value = matrix[i][j]
                if value.denominator == 1:
                    row.append(str(value.numerator))
                else:
                    row.append(f"{value.numerator}/{value.denominator}")
            md.append("| " + " | ".join(row) + " |")
        
        return md

    def _calculate_geometric_mean(self, row: List[Fraction]) -> float:
        """Вычисляет среднее геометрическое для строки матрицы"""
        product = 1.0
        for value in row:
            product *= float(value)
        return math.pow(product, 1.0 / len(row))

    def _calculate_priority_vector(self, matrix: List[List[Fraction]]) -> List[float]:
        """Вычисляет вектор приоритетов ля матрицы"""
        # Вычисляем среднее геометрическое для каждой с��роки
        geometric_means = [self._calculate_geometric_mean(row) for row in matrix]
        
        # Нормализуем значения
        total = sum(geometric_means)
        return [mean/total for mean in geometric_means]

    def calculate_all_priorities(self) -> Dict[str, List[float]]:
        """Вычисляет все векторы приоритетов"""
        priorities = {
            'criteria': self._calculate_priority_vector(self.criteria_matrix)
        }
        
        # Вычисляем векторы приоритетов для каждого критерия
        for criterion in self.criteria:
            priorities[criterion] = self._calculate_priority_vector(
                self.alternatives_matrices[criterion]
            )
            
        return priorities

    def _calculate_lambda_max(self, matrix: List[List[Fraction]], priority_vector: List[float]) -> float:
        """Вычисляет максимальное собственное значение (λmax) матрицы"""
        n = len(matrix)
        # Умножаем матрицу на вектор приоритетов
        weighted_sum_vector = []
        for i in range(n):
            row_sum = 0
            for j in range(n):
                row_sum += float(matrix[i][j]) * priority_vector[j]
            weighted_sum_vector.append(row_sum)
        
        # Вычисляем λmax как среднее отношение компонент векторов
        lambda_max = 0
        for i in range(n):
            lambda_max += weighted_sum_vector[i] / priority_vector[i]
        return lambda_max / n

    def _calculate_consistency(self, matrix: List[List[Fraction]], priority_vector: List[float]) -> Tuple[float, float, float]:
        """Вычисляет λmax, ИС и ОС для матрицы"""
        n = len(matrix)
        lambda_max = self._calculate_lambda_max(matrix, priority_vector)
        
        # Вычисляем индекс согласованности (ИС)
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0
        
        # Вычисляем отношение согласованности (ОС)
        cr = ci / self.RANDOM_CONSISTENCY[n] if n > 2 else 0
        
        return lambda_max, ci, cr

    def calculate_global_priorities(self) -> Dict[str, float]:
        """Вычисляет глобальные приоритеты альтернатив"""
        priorities = self.calculate_all_priorities()
        criteria_priorities = priorities['criteria']
        
        # Инициализируем глобальные приоритеты
        global_priorities = {alt: 0.0 for alt in self.alternatives}
        
        # Для каждой альтернативы
        for i, alternative in enumerate(self.alternatives):
            # Для каждого критерия
            for j, criterion in enumerate(self.criteria):
                # Умножаем локальный приоритет на вес критерия
                global_priorities[alternative] += (
                    priorities[criterion][i] * criteria_priorities[j]
                )
        
        return global_priorities

    def analyze_all_matrices(self) -> Dict[str, Dict[str, float]]:
        """Анализирует согласованность всех матриц"""
        priorities = self.calculate_all_priorities()
        results = {}
        
        # Анализ матрицы критериев
        lambda_max, ci, cr = self._calculate_consistency(
            self.criteria_matrix,
            priorities['criteria']
        )
        results['Критерии'] = {
            'lambda_max': lambda_max,
            'CI': ci,
            'CR': cr
        }
        
        # Анлиз матриц альтернатив
        for criterion in self.criteria:
            lambda_max, ci, cr = self._calculate_consistency(
                self.alternatives_matrices[criterion],
                priorities[criterion]
            )
            results[criterion] = {
                'lambda_max': lambda_max,
                'CI': ci,
                'CR': cr
            }
        
        return results

    def generate_markdown(self) -> str:
        md = []
        
        # ... (предыдущий код для иерархии) ...
        md.append("# Анализ выбора дома по методу Саати\n")
        md.append("## Иерархия для выбора варианта\n")
        
        md.append("### Уровень 1\n")
        md.append("**Дом**\n\n")
        
        md.append("### Уровень 2\n")
        md.append("Критерии оценки:\n")
        for i, criterion in enumerate(self.criteria, 1):
            md.append(f"{i}. {criterion}")
        md.append("\n")
        
        md.append("### Уровень 3\n")
        md.append("Альтернативы:\n")
        for alt in self.alternatives:
            md.append(f"- {alt}")
        md.append("\n")

        # Марица парных сравнений критериев
        md.append("## Матрица парных сравнений критериев\n")
        md.append("### Общее удовлетворение домом\n\n")
        md.extend(self._format_matrix_as_table(self.criteria, self.criteria_matrix))
        md.append("\n")

        # Матрицы парных сравнений альтернатив
        md.append("## Матрицы парных сравнений альтернатив\n")
        for criterion in self.criteria:
            md.append(f"### {criterion}\n")
            md.extend(self._format_matrix_as_table(
                self.alternatives,
                self.alternatives_matrices[criterion]
            ))
            md.append("\n")

        # Добавляем подробный анализ матрицы критериев
        md.append("## Подробный анализ матрицы критериев\n")
        
        # 1. Вычисление среднего геометрического
        md.append("### 1. Вычисление среднего геометрического для каждой строки\n")
        geometric_means = []
        for i, criterion in enumerate(self.criteria):
            row = self.criteria_matrix[i]
            product = " × ".join([f"({float(x):.4f})" for x in row])
            mean = self._calculate_geometric_mean(row)
            geometric_means.append(mean)
            md.append(f"**{criterion}**:")
            md.append(f"* Произведение элементов = {product}")
            md.append(f"* Среднее геометрическое = ({product})^(1/{len(row)}) = {mean:.4f}\n")
        
        # 2. Вычисление вектора приоритетов
        total_mean = sum(geometric_means)
        md.append("### 2. Нормализация и получение вектора приоритетов\n")
        md.append(f"Сумма средних геометрических = {total_mean:.4f}\n")
        md.append("| Критерий | Формула | Значение приоритета |")
        md.append("|----------|---------|-------------------|")
        priorities = []
        for i, criterion in enumerate(self.criteria):
            priority = geometric_means[i] / total_mean
            priorities.append(priority)
            md.append(f"| {criterion} | {geometric_means[i]:.4f} / {total_mean:.4f} | {priority:.4f} |")
        md.append("\n")
        
        # 3. Вычисление λmax
        md.append("### 3. Вычисление λmax\n")
        md.append("#### 3.1 Умножение матрицы на вектор приоритетов:\n")
        weighted_sums = []
        for i, criterion in enumerate(self.criteria):
            row_sum = 0
            calculation = []
            for j, value in enumerate(self.criteria_matrix[i]):
                product = float(value) * priorities[j]
                row_sum += product
                calculation.append(f"({float(value):.4f} × {priorities[j]:.4f})")
            weighted_sums.append(row_sum)
            md.append(f"**{criterion}**: {' + '.join(calculation)} = {row_sum:.4f}\n")
        
        md.append("#### 3.2 Вычисление отношений и λmax:\n")
        ratios = []
        for i, (ws, priority) in enumerate(zip(weighted_sums, priorities)):
            ratio = ws / priority
            ratios.append(ratio)
            md.append(f"* {self.criteria[i]}: {ws:.4f} / {priority:.4f} = {ratio:.4f}")
        
        lambda_max = sum(ratios) / len(ratios)
        md.append(f"\nλmax = ({' + '.join([f'{x:.4f}' for x in ratios])}) / {len(ratios)} = {lambda_max:.4f}\n")
        
        # 4. Вычисление ИС и ОС
        n = len(self.criteria)
        ci = (lambda_max - n) / (n - 1)
        cr = ci / self.RANDOM_CONSISTENCY[n]
        
        md.append("### 4. Вычисление индекса согласованности (ИС) и отношения согласованности (ОС)\n")
        md.append(f"ИС = (λmax - n) / (n - 1) = ({lambda_max:.4f} - {n}) / ({n} - 1) = {ci:.4f}\n")
        md.append(f"ОС = ИС / {self.RANDOM_CONSISTENCY[n]} = {ci:.4f} / {self.RANDOM_CONSISTENCY[n]} = {cr:.4f}\n")
        
        if cr < 0.1:
            md.append("**Матрица согласована** (ОС < 0.1)")
        else:
            md.append("**Матрица не согласована** (ОС ≥ 0.1)")
        md.append("\n")

        # 5. Глобальные приоритеты с подробными вычислениями
        md.append("## Вычисление глобальных приоритетов\n")
        priorities = self.calculate_all_priorities()
        criteria_priorities = priorities['criteria']
        
        for alt_idx, alternative in enumerate(self.alternatives):
            md.append(f"\n### Расчет для {alternative}:\n")
            calculations = []
            total = 0
            
            for crit_idx, criterion in enumerate(self.criteria):
                local_priority = priorities[criterion][alt_idx]
                criterion_weight = criteria_priorities[crit_idx]
                product = local_priority * criterion_weight
                total += product
                
                calculations.append(
                    f"* {criterion}: {local_priority:.4f} × {criterion_weight:.4f} = {product:.4f}"
                )
            
            md.append("Формула: Σ(локальный приоритет × вес критерия)")
            md.extend(calculations)
            md.append(f"\n**Итоговый глобальный приоритет = {total:.4f}**\n")

        # Анализ согласованности всех матриц
        md.append("## Анализ согласованности всех матриц\n")
        consistency_results = self.analyze_all_matrices()
        
        md.append("| Матрица | λmax | ИС | ОС | Статус |")
        md.append("|---------|------|----|----|---------|")
        
        for matrix_name, results in consistency_results.items():
            status = "✓" if results['CR'] < 0.1 else "✗"
            md.append(
                f"| {matrix_name} | {results['lambda_max']:.4f} | "
                f"{results['CI']:.4f} | {results['CR']:.4f} | {status} |"
            )
        md.append("\n")

        # Глобальные приоритеты
        md.append("## Глобальные приоритеы альтернатив\n")
        global_priorities = self.calculate_global_priorities()
        
        md.append("| Альтернатива | Глобальный приоритет |")
        md.append("|--------------|---------------------|")
        for alt, priority in sorted(
            global_priorities.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            md.append(f"| {alt} | {priority:.4f} |")
        md.append("\n")

        # Определение лучшей альтернативы
        best_alternative = max(global_priorities.items(), key=lambda x: x[1])
        md.append(f"**Лучшая альтернатива:** {best_alternative[0]} "
                  f"(приоритет: {best_alternative[1]:.4f})")
        
        # Добавляем раздел с формулами
        md.append("## Формулы метода анализа иерархий\n")

        # 1. Среднее геометрическое
        md.append("### Среднее геометрическое\n")
        md.append("```math")
        md.append(r"\bar{a_i} = \sqrt[n]{\prod_{j=1}^n a_{ij}}")
        md.append("```")
        md.append("\nгде:\n")
        md.append("- $\\bar{a_i}$ - среднее геометрическое для i-й строки")
        md.append("- $a_{ij}$ - элемент матрицы")
        md.append("- $n$ - размерность матрицы\n")

        # 2. Вектор приоритетов
        md.append("### Вектор приоритетов\n")
        md.append("```math")
        md.append(r"w_i = \frac{\bar{a_i}}{\sum_{k=1}^n \bar{a_k}}")
        md.append("```")
        md.append("\nгде:\n")
        md.append("- $\\overline{a}_i$ - среднее геометрическое для i-й строки")
        md.append("- $\\overline{a}_k$ - среднее геометрическое k-й строки")
        md.append("- $n$ - размерность матрицы\n")

        # 3. Вычисление λmax
        md.append("### Максимальное собственное значение (λmax)\n")
        md.append("```math")
        md.append(r"\lambda_{max} = \frac{1}{n}\sum_{i=1}^n \frac{(A\mathbf{w})_i}{w_i}")
        md.append("```")
        md.append("\nгде:\n")
        md.append("- $A$ - матрица парных сравнений")
        md.append("- $\\\mathbf{w}$ - вектор приоритетов")
        md.append("- $(A\\mathbf{w})_i$ - i-я компонента произведения матрицы на вектор приоритетов\n")

        # 4. Индекс согласованности
        md.append("### Индекс согласованности (ИС)\n")
        md.append("```math")
        md.append(r"ИС = \frac{\lambda_{max} - n}{n-1}")
        md.append("```")
        md.append("\nгде:\n")
        md.append("- $\\\lambda_{max}$ - максимальное собственное значение")
        md.append("- $n$ - размерность матрицы\n")

        # 5. Отношение согласованности
        md.append("### Отношение согласованности (ОС)\n")
        md.append("```math")
        md.append(r"ОС = \frac{ИС}{СС}")
        md.append("```")
        md.append("\nгде:\n")
        md.append("- $ИС$ - индекс согласованности")
        md.append("- $СС$ - случайная согласованность (табличное значение)\n")

        # 6. Глобальные приоритеты
        md.append("### Глобальные приоритеты\n")
        md.append("```math")
        md.append(r"GP_i = \sum_{j=1}^m w_j \cdot LP_{ij}")
        md.append("```")
        md.append("\nгде:\n")
        md.append("- $GP_i$ - глобальный приоритет i-й альтернативы")
        md.append("- $w_j$ - вес j-го критерия")
        md.append("- $LP_{ij}$ - локальный приоритет i-й альтернативы по j-му критерию")
        md.append("- $m$ - количество критериев\n")

        # Таблица случайной согласованности
        md.append("### Таблица случайной согласованности\n")
        md.append("| Размер матрицы | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |")
        md.append("|----------------|---|---|---|---|---|---|---|---|---|-----|")
        md.append("| Случайная согласованность | 0 | 0 | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 | 1.45 | 1.49 |")
        md.append("\n")

        return "\n".join(md)

    def save_report(self, filename: str = "house_hierarchy.md"):
        """Сохраняет отчет в markdown-файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.generate_markdown())

    def visualize_results(self, filename_prefix: str = "ahp_visualization"):
        """Создает визуализации результатов анализа"""
        priorities = self.calculate_all_priorities()
        global_priorities = self.calculate_global_priorities()
        
        # График глобальных приоритетов
        plt.figure(figsize=(10, 6))
        alternatives = list(global_priorities.keys())
        values = list(global_priorities.values())
        
        plt.bar(alternatives, values)
        plt.title('Глобальные приоритеты альтернатив')
        plt.ylabel('Приоритет')
        plt.ylim(0, max(values) * 1.1)
        
        for i, v in enumerate(values):
            plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.savefig(f'{filename_prefix}_global.png')
        plt.close()
        
        # График локальных приоритетов по критериям
        plt.figure(figsize=(12, 8))
        x = np.arange(len(self.criteria))
        width = 0.25
        
        for i, alt in enumerate(self.alternatives):
            alt_priorities = [priorities[criterion][i] for criterion in self.criteria]
            plt.bar(x + i*width, alt_priorities, width, label=alt)
        
        plt.xlabel('Критерии')
        plt.ylabel('Локальный приоритет')
        plt.title('Локальные приоритеты по критериям')
        plt.xticks(x + width, self.criteria, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f'{filename_prefix}_local.png')
        plt.close()

def main():
    ahp = AHPHouse()
    
    # Заполнение матрицы парных сравнений критериев
    comparisons = [
        (0, 1, "5"), (0, 2, "3"), (0, 3, "7"), (0, 4, "6"), 
        (0, 5, "6"), (0, 6, "1/3"), (0, 7, "1/4"),
        (1, 2, "1/3"), (1, 3, "5"), (1, 4, "3"),
        (1, 5, "3"), (1, 6, "1/5"), (1, 7, "1/7"),
        (2, 3, "6"), (2, 4, "3"), (2, 5, "4"),
        (2, 6, "6"), (2, 7, "1/5"),
        (3, 4, "1/3"), (3, 5, "1/4"), (3, 6, "1/7"),
        (3, 7, "1/8"),
        (4, 5, "1/2"), (4, 6, "1/5"), (4, 7, "1/6"),
        (5, 6, "1/5"), (5, 7, "1/6"),
        (6, 7, "1/2"),
    ]
    
    for i, j, value in comparisons:
        ahp.set_criteria_comparison(i, j, value)

    # Размеры дома
    ahp.set_alternative_comparison("Размеры дома", 0, 1, "6")
    ahp.set_alternative_comparison("Размеры дома", 0, 2, "8")
    ahp.set_alternative_comparison("Размеры дома", 1, 2, "4")

    # Удобство автобусных маршрутов
    ahp.set_alternative_comparison("Удобство автобусных маршрутов", 0, 1, "7")
    ahp.set_alternative_comparison("Удобство автобусных маршрутов", 0, 2, "1/5")
    ahp.set_alternative_comparison("Удобство автобусных маршрутов", 1, 2, "1/8")
    ahp.alternatives_matrices["Удобство автобусных маршрутов"][2][0] = Fraction(5)  # Исправляем значение
    ahp.alternatives_matrices["Удобство автобусных маршрутов"][2][1] = Fraction(8)  # Исправляем значение

    # Окрестности
    ahp.set_alternative_comparison("Окрестности", 0, 1, "8")
    ahp.set_alternative_comparison("Окрестности", 0, 2, "6")
    ahp.set_alternative_comparison("Окрестности", 1, 2, "1/4")
    ahp.alternatives_matrices["Окрестности"][2][1] = Fraction(4)  # Исправляем значение

    # Когда построен дом
    ahp.set_alternative_comparison("Когда построен дом", 0, 1, "1")
    ahp.set_alternative_comparison("Когда построен дом", 0, 2, "1")
    ahp.set_alternative_comparison("Когда построен дом", 1, 2, "1")

    # Двор
    ahp.set_alternative_comparison("Двор", 0, 1, "5")
    ahp.set_alternative_comparison("Двор", 0, 2, "4")
    ahp.set_alternative_comparison("Двор", 1, 2, "1/3")
    ahp.alternatives_matrices["Двор"][2][1] = Fraction(3)  # Исправляем значение

    # Современное оборудование
    ahp.set_alternative_comparison("Современное оборудование", 0, 1, "8")
    ahp.set_alternative_comparison("Современное оборудование", 0, 2, "6")
    ahp.set_alternative_comparison("Современное оборудование", 1, 2, "1/5")
    ahp.alternatives_matrices["Современное оборудование"][2][1] = Fraction(5)  # Исправляем значение

    # Общее состояние
    ahp.set_alternative_comparison("Общее состояние", 0, 1, "1/2")
    ahp.set_alternative_comparison("Общее состояние", 0, 2, "1/2")
    ahp.alternatives_matrices["Общее состояние"][1][0] = Fraction(2)  # Исправляем значение
    ahp.alternatives_matrices["Общее состояние"][2][0] = Fraction(2)  # Исправляем значение
    ahp.set_alternative_comparison("Общее состояние", 1, 2, "1")

    # Финансовые условия
    ahp.set_alternative_comparison("Финансовые условия", 0, 1, "1/7")
    ahp.set_alternative_comparison("Финансовые условия", 0, 2, "1/5")
    ahp.alternatives_matrices["Финансовые условия"][1][0] = Fraction(7)  # Исправляем значение
    ahp.alternatives_matrices["Финансовые условия"][2][0] = Fraction(5)  # Исправляем значение
    ahp.set_alternative_comparison("Финансовые условия", 1, 2, "3")
    
    # Сохраняем отчет и визуализации
    ahp.save_report()
    ahp.visualize_results()

if __name__ == '__main__':
    main()