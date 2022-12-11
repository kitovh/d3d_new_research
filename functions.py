### Repository of functions to use for D3D research
from scipy.interpolate import interp1d
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy.io

def load_and_filter_data(file_name, output_columns=False, Filter=True):
    """
    Load .mat data file, and if needed save columns to excel
    
    Inputs:
        file_name: .mat file
        output_columns (Optional): bool, will output columns.csv with names of data columns 
    """
    
    mat = scipy.io.loadmat(file_name)
    mat = dict((k, mat[k]) for k in mat.keys() if (k != '__header__') & (k != '__version__') & (k != '__globals__'))

    if Filter:
        flattened_values = [mat[list(mat.keys())[i]].flatten() for i in range(len(list(mat.keys())))]
        
        keys = list(mat.keys())
        mat = dict(zip(keys, flattened_values))
        new_mat = {}
        for (k, v) in mat.items():
            if len(v) == 3018096:
                new_mat[k] = v

        diii_d_df = pd.DataFrame(new_mat)

        if output_columns:        
            original_columns = pd.DataFrame(mat.keys())
            original_columns.columns = ['variable_name']
            original_columns['origins'] = 'original'

            filtered_columns = pd.DataFrame(list(diii_d_df.columns))
            filtered_columns.columns = ['variable_name']
            filtered_columns['origins'] = 'filtered'

            missing_columns = list(set(original_columns['variable_name']).difference(filtered_columns['variable_name']))
            df = pd.concat([filtered_columns, pd.DataFrame({'variable_name':missing_columns, 'origins':'original'})])

            df.to_csv('columns.csv')
    
        return mat, diii_d_df
    else:
        return mat
    
    
def _update_state(A, W_in, state, inputs, symetry=False):
    
    """
    Computes the next network states by applying the recurrent weights
    to the last state & and feeding in the current input patterns
    Following r(t+1) = tanh{A * r(t) + W_in * u(t)}
    
    Args:
    ----
        state: The preview states.
        input_pattern: Next Intputs
    """
    
    NextState = np.tanh(A @ state + W_in @ inputs)
    NextState = np.asarray(np.matrix(NextState))
    
    if symetry is True:
        AugmentedNextState = NextState.copy()
        AugmentedNextState[::2] = NextState[::2] ** 2
        
        return(NextState, AugmentedNextState)
    else:
        return(NextState, None)
    
    
def interpolate(data, input_rolling_window=10, granularity=0.0002):
    
    interpolated_data = []
    for shot_number in tqdm(data['shot'].unique()):
        shot_data_backup = data[data['shot'] == shot_number]

        ## granularity = 0.0002

        x = np.arange(shot_data_backup['time'].min(), shot_data_backup['time'].max(), granularity)
        x = x[x <= shot_data_backup['time'].max()]

        f1 = interp1d(shot_data_backup['time'], shot_data_backup['ip'])

        interpolated = pd.DataFrame({'time':x, 
                                    'ip':f1(x)})
        interpolated.set_index('time', inplace=True)

        ## Lag by 40ms = 0.04 seconds 
        output_shift = int(0.04 / granularity)
        interpolated.columns = ['input 0']

        values = list(range(1, input_rolling_window+1))

        for a in values:
            interpolated['input ' + str(a)] = interpolated[['input 0']].shift(-a)
            if a == max(values):
                interpolated['output'] = interpolated[['input ' + str(a)]].shift(-output_shift)

        interpolated['shot']  = shot_number

        interpolated_data.append(interpolated)

    interpolated_data = pd.concat(interpolated_data)
    
    return(interpolated_data)


def test_on_data(testing_inputs, testing_outputs, r_matrix, W_in, W_out, ReservoirSize, A):
    testing_r_matrix = np.zeros((testing_inputs.shape[0], ReservoirSize))
    testing_r_matrix[0] = r_matrix
    
    for t in tqdm(range(0, testing_inputs.shape[0])):
        testing_r_matrix[t, :], _ = _update_state(
            A=A,
            W_in=W_in,
            state=testing_r_matrix[t],
            inputs=testing_inputs[t, :],
            symetry=False
        )
    
    PredictedTesting = np.dot(testing_r_matrix, W_out)
    return PredictedTesting[1:], testing_outputs[:-1]
