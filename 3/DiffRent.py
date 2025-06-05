import numpy as np

def differential_rents_method(costs, supply, demand):
    costs = np.array(costs, dtype=float)
    supply = np.array(supply, dtype=float)
    demand = np.array(demand, dtype=float)

    if not np.isclose(supply.sum(), demand.sum()):
        raise ValueError("Несбалансированная задача!")

    m, n = len(supply), len(demand)
    plan = np.zeros((m, n))
    steps = []

    # === Шаг 1. Построение начального плана ===
    for j in range(n):
        min_cost = float('inf')
        index = -1
        for i in range(m):
            if costs[i, j] < min_cost:
                min_cost = costs[i, j]
                index = i
        plan[index, j] = demand[j]
        steps.append((f"Начальное распределение: ячейка ({index + 1},{j + 1})", plan.copy(), [(index, j)]))

    # === Шаг 2. Перераспределение ===
    while True:
        row_sums = plan.sum(axis=1)
        row_differences = supply - row_sums

        if all(diff >= 0 for diff in row_differences):
            break

        i_minus = next(i for i, diff in enumerate(row_differences) if diff < 0)
        i_plus_candidates = [i for i, diff in enumerate(row_differences) if diff > 0]

        min_diff = float('inf')
        best_j = -1
        best_i_plus = -1
        for j in range(n):
            if plan[i_minus, j] > 0:
                for i_plus in i_plus_candidates:
                    diff = costs[i_plus, j] - costs[i_minus, j]
                    if diff < min_diff:
                        min_diff = diff
                        best_j = j
                        best_i_plus = i_plus

        if best_j != -1 and best_i_plus != -1:
            qty = min(-row_differences[i_minus], row_differences[best_i_plus], plan[i_minus, best_j])
            plan[i_minus, best_j] -= qty
            plan[best_i_plus, best_j] += qty
            steps.append(
                (f"Перераспределение: из ({i_minus + 1},{best_j + 1}) → ({best_i_plus + 1},{best_j + 1}) на {qty}",
                 plan.copy(),
                 [(i_minus, best_j), (best_i_plus, best_j)]))

    total_cost = np.sum(plan * costs)
    return plan, total_cost, steps


def print_plan_pandas(plan, supply, demand, highlights=None):
    highlights = highlights or []
    m, n = plan.shape

    df = pd.DataFrame(plan.astype(int),
                      index=[f"A{i + 1}" for i in range(m)],
                      columns=[f"B{j + 1}" for j in range(n)])

    df['Поставка'] = supply.astype(int)
    demand_row = [f"{int(val)}" for val in demand]
    demand_row += ['']
    demand_df = pd.DataFrame([demand_row], columns=df.columns, index=['Потребность'])

    # Подсвечивание выделенных ячеек (как строка с *)
    def highlight(val, i, j):
        return f"*{val}" if (i, j) in highlights else str(val)

    styled_df = df.copy()
    for i in range(m):
        for j in range(n):
            styled_df.iloc[i, j] = highlight(df.iloc[i, j], i, j)

    result = pd.concat([styled_df, demand_df])
    print(result.to_string(index=True))
    print()


def output(supply, demand, plan, steps):
    print("\nШаги построения плана:")
    for title, matrix, highlights in steps:
        print(f"\n{title}")
        print_plan_pandas(matrix, supply, demand, highlights)

    print("Финальный план перевозок:")
    print_plan_pandas(plan, supply, demand)


if __name__ == "__main__":
    print("Пример 1")
    supply = [180, 350, 20]
    demand = [110, 90, 120, 80, 150]
    costs = [
        [1, 12, 4, 8, 5],
        [7, 8, 6, 5, 3],
        [6, 13, 8, 7, 4]
    ]

    plan, total_cost, steps = differential_rents_method(costs, supply, demand)
    output(supply, demand, plan, steps)
    print(f"Минимальная суммарная стоимость: {total_cost:.2f}")
