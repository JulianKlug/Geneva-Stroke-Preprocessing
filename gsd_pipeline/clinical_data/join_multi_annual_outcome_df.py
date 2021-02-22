import os
import pandas as pd



def join_multi_annual_outcome_df(df_path_1, df_path_2, pid_df1: str, pid_df2: str, outcomes_df1: list,
                                 outcomes_df2: list, output_dir: str = None):
    '''
    Returns a dataframe saving a combination of both dataframes with specified outcomes

    Example:
        join_multi_annual_outcome_df('/path',
                             '/path',
                             'anonymised_id',
                             'pid',
                             ['combined_mRS_90_days'],
                             ['3M mRS'])
    '''

    df1 = pd.read_excel(df_path_1)
    df2 = pd.read_excel(df_path_2)

    outcome_df = df1[[pid_df1] + outcomes_df1]

    df2[pid_df1] = df2[pid_df2]
    df2[outcomes_df1] = df2[outcomes_df2]

    outcome_df = outcome_df.append(df2[[pid_df1] + outcomes_df1])

    if output_dir is None:
        output_dir = os.path.dirname(df_path_1)

    outcome_df.to_excel(os.path.join(output_dir, 'joined_anon_outcome_df.xlsx'))
