import numpy as np
import tensorflow as tf
import pandas as pd

from time import time
from statistics import mean, stdev

from milp import codify_network

# For type annotations
from docplex.mp.model import Model
from docplex.mp.constr import LinearConstraint
from docplex.mp.dvar import Var
from tensorflow.python.framework.ops import EagerTensor


def insert_output_constraints_fischetti(
        mp_model: Model,
        output_variables: list,
        network_output: EagerTensor,
        binary_variables: list[Var]
) -> Model:
    variable_output: Var  = output_variables[network_output]  # variable of name f'o_{network_output}'

    aux_var = 0
    for i, output in enumerate(output_variables):
        if i != network_output:
            p = binary_variables[aux_var]
            aux_var += 1
            mp_model.add_indicator(binary_var=p, linear_ct=variable_output <= output, active_value=1)

    return mp_model


def insert_output_constraints_tjeng(
        mp_model: Model,
        output_variables: list,
        network_output: EagerTensor,
        binary_variables: list[Var],
        output_bounds: list[list[float]]
) -> Model:
    assert output_bounds is not None and output_bounds != [], 'If the method tjeng is chosen, output_bounds must be passed.'

    variable_output: Var | None = output_variables[network_output]
    upper_bounds_diffs = output_bounds[network_output][1] - np.array(output_bounds)[:, 0]  # Output i: oi - oj <= u1 = ui - lj
    aux_var = 0
    for i, output in enumerate(output_variables):
        if i != network_output:
            ub = upper_bounds_diffs[i]
            z = binary_variables[aux_var]
            mp_model.add_constraint(variable_output - output - ub * (1 - z) <= 0)
            aux_var += 1

    return mp_model


def get_minimal_explanation(
        mp_model: Model,
        network_input: EagerTensor,
        network_output: EagerTensor,
        n_classes: int,
        method: str,
        output_bounds: list[list[float]] = None
) -> list[LinearConstraint]:
    # `network_output` is the predicted value

    mp_model = insert_output_constraints(mp_model, method, n_classes, network_output, output_bounds)

    input_variables = [mp_model.get_var_by_name(f'x_{i}') for i in range(len(network_input[0]))]
    input_constraints = mp_model.add_constraints(
        [input_variables[i] == feature.numpy() for i, feature in enumerate(network_input[0])],
        names='input'
    )

    # Filter relevant features (i.e. features that are important for the classification)
    for i in range(len(network_input[0])):
        constraint = input_constraints[i]
        mp_model.remove_constraint(constraint)

        mp_model.solve(log_output=False)

        if mp_model.solution is not None:  # "If the prediction wouldn't change"
            mp_model.add_constraint(constraint)

    return mp_model.find_matching_linear_constraints('input')


def insert_output_constraints(
        mp_model: Model,
        method: str,
        n_classes: int,
        network_output: EagerTensor,
        output_bounds: list[list[float]]
) -> Model:
    output_variables: list = [mp_model.get_var_by_name(f'o_{i}') for i in range(n_classes)]
    binary_variables: list[Var] = mp_model.binary_var_list(n_classes - 1, name='b')
    # Indicates that at least 1 of the binary variables is true
    mp_model.add_constraint(mp_model.sum(binary_variables) >= 1)
    if method == 'tjeng':
        mp_model = insert_output_constraints_tjeng(mp_model, output_variables, network_output,
                                                   binary_variables, output_bounds)
    else:
        mp_model = insert_output_constraints_fischetti(mp_model, output_variables, network_output,
                                                       binary_variables)

    return mp_model


def main():
    datasets_info = [
        # {'name': 'australian', 'n_classes': 2},
        # {'name': 'auto', 'n_classes': 5},
        # {'name': 'backache', 'n_classes': 2},
        # {'name': 'breast-cancer', 'n_classes': 2},
        # {'name': 'cleve', 'n_classes': 2},
        # {'name': 'cleveland', 'n_classes': 5},
        {'name': 'glass', 'n_classes': 5},
        # {'name': 'glass2', 'n_classes': 2},
        # {'name': 'heart-statlog', 'n_classes': 2},
        # {'name': 'hepatitis', 'n_classes': 2},
        # {'name': 'spect', 'n_classes': 2},
        # {'name': 'voting', 'n_classes': 2}
    ]

    configurations = [
        # {'method': 'fischetti', 'relaxe_constraints': True},
        {'method': 'fischetti', 'relaxe_constraints': False},
        # {'method': 'tjeng', 'relaxe_constraints': True},
        # {'method': 'tjeng', 'relaxe_constraints': False}
    ]

    df = {
        'fischetti': {
            True: {'size': [], 'milp_time': [], 'build_time': []},
            False: {'size': [], 'milp_time': [], 'build_time': []}
        },
        'tjeng': {
            True: {'size': [], 'milp_time': [], 'build_time': []},
            False: {'size': [], 'milp_time': [], 'build_time': []}
        }
    }

    for dataset_info in datasets_info:
        dataset_name = dataset_info['name']
        n_classes = dataset_info['n_classes']

        for configuration in configurations:
            print(dataset_info, configuration)

            method = configuration['method']
            relaxe_constraints = configuration['relaxe_constraints']

            data_test = pd.read_csv(f'datasets/{dataset_name}/test.csv')
            data_train = pd.read_csv(f'datasets/{dataset_name}/train.csv')

            dataframe = data_train.append(data_test)

            model_path = f'datasets/{dataset_name}/model_2layers_{dataset_name}.h5'
            keras_model = tf.keras.models.load_model(model_path)

            network_codifying_times = []

            start = time()
            mp_model, output_bounds = codify_network(keras_model, dataframe, method, relaxe_constraints)
            network_codifying_times.append(time() - start)
            print('Network codifying time:', network_codifying_times[-1])

            minimal_explanation_times = []
            explanation_lengths = []
            data = dataframe.to_numpy()

            for _ in range(1):
                i = 127
                # shape[0]: number of rows
                # shape[1]: number of columns

                # if i % 50 == 0:
                print(i)
                network_input = data[i, :-1]  # `network_input` doesn't change...

                network_input = tf.reshape(tf.constant(network_input), (1, -1))
                network_output = keras_model.predict(tf.constant(network_input))[
                    0]  # ...therefore, `network_output` also doesn't change
                network_output = tf.argmax(network_output)

                # `network_input` is constant
                # Therefore, `network_output` is also constant

                start = time()
                minimal_explanation = get_minimal_explanation(mp_model, network_input, network_output,
                                                              n_classes, method, output_bounds)

                print(mp_model.lp_string)

                minimal_explanation_times.append(time() - start)

                explanation_lengths.append(len(minimal_explanation))

                print('Minimal explanation:')
                for constraint in minimal_explanation:
                    print(constraint)

            df[method][relaxe_constraints]['size'].extend(
                [min(explanation_lengths), f'{mean(explanation_lengths)} +- {stdev(explanation_lengths)}',
                 max(explanation_lengths)])
            df[method][relaxe_constraints]['milp_time'].extend([min(minimal_explanation_times),
                                                                f'{mean(minimal_explanation_times)} +- {stdev(minimal_explanation_times)}',
                                                                max(minimal_explanation_times)])
            df[method][relaxe_constraints]['build_time'].extend(
                [min(network_codifying_times), f'{mean(network_codifying_times)} +- {stdev(network_codifying_times)}',
                 max(network_codifying_times)])

            print_statistics(explanation_lengths, 'Explanation size')
            print_statistics(minimal_explanation_times, 'Explanation time')
            print_statistics(network_codifying_times, 'Build time')

    df = {
        # 'fischetti_relaxe_size': df['fischetti'][True]['size'],
        # 'fischetti_relaxe_time': df['fischetti'][True]['milp_time'],
        # 'fischetti_relaxe_build_time': df['fischetti'][True]['build_time'],
        'fischetti_not_relaxe_size': df['fischetti'][False]['size'],
        'fischetti_not_relaxe_time': df['fischetti'][False]['milp_time'],
        'fischetti_not_relaxe_build_time': df['fischetti'][False]['build_time'],
        # 'tjeng_relaxe_size': df['tjeng'][True]['size'],
        # 'tjeng_relaxe_time': df['tjeng'][True]['milp_time'],
        # 'tjeng_relaxe_build_time': df['tjeng'][True]['build_time'],
        # 'tjeng_not_relaxe_size': df['tjeng'][False]['size'],
        # 'tjeng_not_relaxe_time': df['tjeng'][False]['milp_time'],
        # 'tjeng_not_relaxe_build_time': df['tjeng'][False]['build_time']
    }

    index_label = []
    for dataset_info in datasets_info:
        index_label.extend([f"{dataset_info['name']}_m", f"{dataset_info['name']}_a", f"{dataset_info['name']}_M"])

    df = pd.DataFrame(data=df, index=index_label)
    df.to_csv('results2.csv')


def print_statistics(list: list[float], title: str):
    print(title + ':')
    print(f'\tm: {min(list)}')
    print(f'\ta: {mean(list)} +- {stdev(list)}')
    print(f'\tM: {max(list)}')


if __name__ == '__main__':
    # cProfile.run('main()', sort='time')
    main()
