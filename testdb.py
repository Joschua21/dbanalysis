import os
import sys
sys.path.insert(0, "/bpodautopy/bpodautopy")

import db
import json
import pandas as pd
import matplotlib.pyplot as plt
dbe = db.Engine()

subjects = ['DIG-R-0015', 'DIG-R-0016', 'DIG-R-0017', 'DIG-R-0018']
thresholds = {
    'DIG-R-0015': 52875,
    'DIG-R-0016': 53036,
    'DIG-R-0017': 53451,
    'DIG-R-0018': 53625
} #thresholds of sessids when animals progressed to protocol. Will exclude pretraining from dataframes_trials

dataframes_sess = {}
dataframes_trials = {}

#Extract variables for sessions of all subjects
for subject in subjects:
    query_sess = f"""
        SELECT x.subjid, x.hits, x.viols, x.num_trials, x.total_profit, x.sessid
        FROM beh.sessview x
        WHERE x.subjid = '{subject}'
        AND x.protocol = 'SoundITI2AFC_fm'
        AND x.sess_min IS NOT NULL
        AND x.sess_min != 0
        ORDER BY x.sessiondate ASC
    """
    dataframes_sess[subject] = pd.read_sql(query_sess, dbe)
#Correction of Values that are 0 because of missing Reward History for one Day (19.11.2024)
dataframes_sess['DIG-R-0015'].iloc[5, dataframes_sess['DIG-R-0015'].columns.get_loc('hits')] = 44.77
dataframes_sess['DIG-R-0016'].iloc[4, dataframes_sess['DIG-R-0016'].columns.get_loc('hits')] = 44.81
dataframes_sess['DIG-R-0017'].iloc[1, dataframes_sess['DIG-R-0017'].columns.get_loc('hits')] = 5.83
dataframes_sess['DIG-R-0018'].iloc[0, dataframes_sess['DIG-R-0018'].columns.get_loc('hits')] = 17.02

#Extracts variables for all trials for each subject
for subject in subjects:
    query_trials = f"""
        SELECT x.subjid, x.hit, x.viol, x.RT, x.n_pokes, x.sessid, x.parsed_events
        FROM beh.trialsview x
        WHERE subjid = '{subject}'
        AND x.sessid >= {thresholds[subject]}
        ORDER BY x.trialtime ASC
    """
    result = pd.read_sql(query_trials, dbe)
    dataframes_trials[subject] = result

#Average #pokes for hit per session
for subject in subjects:
    hit_trials = dataframes_trials[subject][dataframes_trials[subject]['hit'] == 1]
    avg_pokes = hit_trials.groupby('sessid')['n_pokes'].mean().reset_index()

    x_values = range(len(avg_pokes))  # Regularly spaced values for the x-axis
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, avg_pokes['n_pokes'], marker='o')

    plt.xlabel('Session')
    plt.ylabel('Average Number of Pokes for Hit Trials')
    plt.title(f'Average Number of Pokes for Hit Trials - {subject}')

    plt.show()


#Moving Average of accuracy at trial i (cummulative hits up to trial i / i) with flexible window size
#Possibly change start, e.g., weighing in the average accuracy of first session to the cumulative accuracy
window_size = 5  # Total of 5 trials in the moving window (centered)

for subject in subjects:
    trials = dataframes_trials[subject]

    # Calculate cumulative accuracy at each trial
    trials['cumulative_accuracy'] = trials['hit'].expanding().mean()

    # Apply a moving average to the cumulative accuracy
    trials['moving_avg_accuracy'] = (
        trials['cumulative_accuracy']
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
    )

    # Plot the moving average of accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(len(trials)),
        trials['moving_avg_accuracy'],
        marker='o',
        label='Moving Avg Accuracy'
    )
    plt.xlabel('Trials (Relative Index)')
    plt.ylabel('Moving Average of Accuracy')
    plt.title(f'Moving Average of Accuracy Across Trials - {subject}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



#Moving Average with window-size of n-30
for subject in subjects:
    # Set window size conditionally
    if subject in ["DIG-R-0015", "DIG-R-0017"]:
        window_size = 100
    elif subject in ["DIG-R-0016", "DIG-R-0018"]:
        window_size = 40
    else:
        raise ValueError(f"Unknown subject ID: {subject}")

    trials = dataframes_trials[subject]

    # Calculate moving average with the chosen window size
    trials['moving_avg_accuracy'] = (
        trials['hit']
        .rolling(window=window_size, center=False, min_periods=1)
        .mean()
    )

    # Plot the moving average of accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(len(trials)),
        trials['moving_avg_accuracy'] * 100,
        marker='o',
        label=f'Moving Avg Accuracy (Window Size: {window_size})'
    )
    plt.xlabel('Trials (Relative Index)')
    plt.ylabel('Moving Average of Accuracy')
    plt.title(f'Moving Average of Accuracy Across Trials - {subject}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# % of Trials completed per session
trials_per_session_data = {}

for subject in subjects:
    trials = dataframes_trials[subject]  # Trials data for the subject

    # Group by session and calculate the total trials and hits per session
    trials_per_session = trials.groupby('sessid').agg(
        total_trials=('hit', 'count'),  # Total trials in the session
        hits=('hit', 'sum')            # Number of hits in the session
    ).reset_index()

    # Calculate accuracy (hits/total_trials)
    trials_per_session['accuracy'] = trials_per_session['hits'] / trials_per_session['total_trials']

    # Store the result in the dictionary
    trials_per_session_data[subject] = trials_per_session

    # Print for inspection (optional)
    print(f"Subject: {subject}")
    print(trials_per_session.head())

# Example of accessing data for a subject
subject = 'DIG-R-0015'
trials_per_session_data[subject]



# Process parsed_events column
for subject, df in dataframes_trials.items():
    # Deserialize the parsed_events column
    df['parsed_events'] = df['parsed_events'].apply(lambda x: json.loads(x))



#Plotting Hits per Session

plt.figure(figsize=(10, 6))
line_styles = {
    'DIG-R-0015': '-',
    'DIG-R-0016': '--',
    'DIG-R-0017': '-',
    'DIG-R-0018': '--',
}

# Plot each subject's data
for subject, df in dataframes_sess.items():
    session_indices = range(1, len(df) + 1)  # Generate session indices
    plt.plot(
        session_indices,
        df['hits'],
        marker='o',
        linestyle=line_styles[subject],
        label=subject
    )

# Customize the legend order and style
legend_order = ['DIG-R-0015', 'DIG-R-0016', 'DIG-R-0017', 'DIG-R-0018']
handles, labels = plt.gca().get_legend_handles_labels()
sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: legend_order.index(x[1]))
sorted_handles, sorted_labels = zip(*sorted_handles_labels)

# Add labels, legend, and title
plt.xlabel('Session (Relative Index)')
plt.ylabel('Hits')
plt.title('Sessions vs. Hits for All Subjects')
plt.legend(sorted_handles, sorted_labels, title='Subject')
plt.grid(True)
plt.tight_layout()

plt.show()