from typing import List, Tuple

def find_optimal_value_with_counts(max_space: int, w: List[int], price_item: List[float], counts: List[int]) -> Tuple[float, List[int]]:
    num_items = len(w)
    d = [[0.0 for _ in range(max_space + 1)] for _ in range(num_items + 1)]

    used = [[0 for _ in range(max_space + 1)] for _ in range(num_items + 1)]

    for i in range(1, num_items + 1):
        for j in range(max_space + 1):
            d[i][j] = d[i - 1][j]
            for k in range(1, min(counts[i - 1], j // w[i - 1]) + 1):
                new_val = d[i - 1][j - k * w[i - 1]] + k * price_item[i - 1]
                if new_val > d[i][j]:
                    d[i][j] = new_val
                    used[i][j] = k


    selected_items = []
    i, j = num_items, max_space
    while i > 0 and j > 0:
        k = used[i][j]
        if k > 0:
            selected_items.extend([i - 1] * k)
            j -= k * w[i - 1]
        i -= 1

    selected_items.reverse()
    return d[num_items][max_space], selected_items

value, items = find_optimal_value_with_counts(
    10,
    [1, 2, 3, 4,5,6,7,8,9,10],
    [2,5,7,9,12,16,17,18,19,21],
    [9,9,9,9,9,9,9,9,9,9]
)
print("Макс. стоимость:", value)
print("Выбранные предметы (индексы):", items)
