import numpy as NP
import pandas as PD

def generate_random_records():
    size = 1000
    columns = [f'x_{_}' for _ in range(10)]
    final_columns = ['y'] + columns

    final_output = {}
    for single_column_index in range(len(final_columns)):
        final_output.update({
            final_columns[single_column_index]: list(NP.random.random(size) * 100)
        })
    return PD.DataFrame(final_output)

if __name__ == '__main__':
    test_data = generate_random_records()

