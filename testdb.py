import os
import sys
sys.path.insert(0, "C:\\Users\\josch\\PycharmProjects\\myenv1\\bpodautopy\\bpodautopy")

import db
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dbe = db.Engine()

subjects = ['DIG-R-0015', 'DIG-R-0016', 'DIG-R-0017', 'DIG-R-0018']
thresholds = {
    'DIG-R-0015': 52875,
    'DIG-R-0016': 53036,
    'DIG-R-0017': 53451,
    'DIG-R-0018': 53625
} #thresholds of sessids when animals progressed to protocol. Will exclude pretraining from dataframes_trials
color_palette = {
    'DIG-R-0015': {'hit': 'darkblue', 'non_hit': 'lightblue'},
    'DIG-R-0016': {'hit': 'darkorange', 'non_hit': 'lightcoral'},
    'DIG-R-0017': {'hit': 'darkgreen', 'non_hit': 'lightgreen'},
    'DIG-R-0018': {'hit': 'darkred', 'non_hit': 'lightpink'}
}
line_styles = {
    'DIG-R-0015': '-',
    'DIG-R-0016': '--',
    'DIG-R-0017': '-',
    'DIG-R-0018': '--',
}

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
    sessids = dataframes_sess[subject]['sessid']
    query_trials = f"""
        SELECT x.subjid, x.hit, x.viol, x.RT, x.n_pokes, x.sessid, x.parsed_events, x.trialid, x.data
        FROM beh.trialsview x
        WHERE subjid = '{subject}'
        AND x.sessid IN ({', '.join([str(sessid) for sessid in sessids])})
        ORDER BY x.trialtime ASC
    """
    result = pd.read_sql(query_trials, dbe)
    dataframes_trials[subject] = result


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



#Moving Average with window-size of n-50; n-10
def plot_moving_avg_accuracy(dataframes_trials, color_palette):
    for subject in subjects:
        # Set window size conditionally
        if subject in ["DIG-R-0015", "DIG-R-0017"]:
            window_size = 50
        elif subject in ["DIG-R-0016", "DIG-R-0018"]:
            window_size = 10
        else:
            raise ValueError(f"Unknown subject ID: {subject}")

        trials = dataframes_trials[subject]

        # Calculate moving average with the chosen window size
        trials['moving_avg_accuracy'] = (
            trials['hit']
            .rolling(window=window_size, center=False, min_periods=1)
            .mean()
        )

        # Get the subject-specific color
        subject_color = color_palette[subject]['hit']

        # Plot the moving average of accuracy
        plt.figure(figsize=(12, 6))
        plt.plot(
            range(len(trials)),
            trials['moving_avg_accuracy'] * 100,  # Convert to percentage
            color=subject_color,
            label=f'Moving Avg Accuracy (Window Size: {window_size})'
        )

        # Updated styling
        plt.xlabel('Trials (Relative Index)', fontsize=14)
        plt.ylabel('Moving Average of Accuracy (%)', fontsize=14)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Remove top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()
# Call the function
plot_moving_avg_accuracy(dataframes_trials, color_palette)

# Process parsed_events column
def process_parsed_events_hit_trials(data):
    reward_state_count = 0
    drink_state_count = 0
    poke_counts = []

    # Iterate through each trial's parsed_events
    for trial, is_hit in data:
        if is_hit != 1:  # Explicitly check if hit equals 1
            continue

        parsed_events = json.loads(trial)  # Deserialize the JSON

        # 1. Process RewardState and drink_state_in_1
        reward_state = parsed_events['vals']['States'].get('RewardState', [None, None])
        drink_state_in = parsed_events['vals']['States'].get('drink_state_in_1', [None, None])

        if reward_state[0] is not None or reward_state[1] is not None:
            reward_state_count += 1
            if drink_state_in[0] is not None or drink_state_in[1] is not None:
                drink_state_count += 1

        # 2. Process Events to count pokes and extract second value from 'dim__'
        poke_events = {key: value for key, value in parsed_events['vals']['Events'].items() if "In" in key}

        # Count pokes based on 'dim__' value (second value in the 'dim__' list)
        total_pokes = 0
        for key, value in poke_events.items():
            if key in parsed_events['info']['Events'] and "In" in key:
                dim_value = parsed_events['info']['Events'][key]['dim__']
                if len(dim_value) > 1:  # Check if dim__ exists and has multiple values
                    total_pokes += dim_value[1]  # Take the second value from 'dim__'
                else:
                    print("Error: No second value exists")
        poke_counts.append(total_pokes)

    # Calculate the percentage
    drink_percentage = (drink_state_count / reward_state_count) * 100 if reward_state_count > 0 else 0

    return drink_percentage, poke_counts

for subject in subjects:
    # Use the pre-existing dataframes_trials
    result = dataframes_trials[subject]

    # Process only hit trials
    drink_percentage, poke_counts = process_parsed_events_hit_trials(zip(result['parsed_events'], result['hit']))

    print(f"Subject: {subject}")
    print(f"Percentage of trials with drink_state_in_1 given RewardState (for hit trials): {drink_percentage:.2f}%")
    print(f"Total pokes per hit trial: {poke_counts}")

#Distribution of #n Pokes --> see below, not used right now
for subject in subjects:
    # Use the pre-existing dataframes_trials
    data = dataframes_trials[subject]

    # Filter trials based on the threshold for the specific subject
    data_filtered = data[data['sessid'] >= thresholds[subject]]

    # Process only the hit trials and get the poke counts
    drink_percentage, poke_counts = process_parsed_events_hit_trials(
        zip(data_filtered['parsed_events'], data_filtered['hit']))

    # Count the distribution of the number of pokes (value counts)
    poke_count_distribution = pd.Series(poke_counts).value_counts().sort_index()

    # Plot the distribution as a histogram
    plt.figure(figsize=(8, 6))
    poke_count_distribution.plot(kind='bar', color='skyblue')

    plt.title(f'Distribution of Pokes per Hit Trial for Subject {subject}')
    plt.xlabel('Number of Pokes')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()


#Plotting Hits per Session
legend_order = ['DIG-R-0015', 'DIG-R-0016', 'DIG-R-0017', 'DIG-R-0018']
def plot_sessions_vs_hits(dataframes_sess, color_palette, line_styles, legend_order):
    plt.figure(figsize=(12, 6))

    for subject, df in dataframes_sess.items():
        session_indices = range(1, len(df) + 1)  # Generate session indices
        subject_color = color_palette[subject]['hit']
        plt.plot(
            session_indices,
            df['hits'],
            marker='o',
            color=subject_color,
            linestyle=line_styles[subject],
            label=subject
        )

    # Customize the legend order and style
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: legend_order.index(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)

    # Add labels, legend, and title
    plt.xlabel('Session', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(sorted_handles, sorted_labels, title='Subject', fontsize=14)

    # Styling modifications
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

plot_sessions_vs_hits(dataframes_sess, color_palette, line_styles, legend_order)

#Poke identities and distribution for all trials (separated into hit/no hit)
def process_parsed_events_all_trials(data, hit_only=True):
    poke_identities = {
        'MidRIn': 0, 'TopRIn': 0, 'BotRIn': 0,
        'MidCIn': 0, 'BotCIn': 0,
        'MidLIn': 0, 'TopLIn': 0, 'BotLIn': 0
    }
    poke_counts = []

    # Iterate through each trial's parsed_events
    for trial, is_hit in data:
        if hit_only and is_hit != 1:
            continue
        elif not hit_only and is_hit != 0:
            continue

        parsed_events = json.loads(trial)  # Deserialize the JSON

        # Process Events to count pokes and track identities
        poke_events = {key: value for key, value in parsed_events['vals']['Events'].items() if "In" in key}

        # Count pokes and track identities
        total_pokes = 0
        for key, value in poke_events.items():
            if key in parsed_events['info']['Events'] and "In" in key:
                dim_value = parsed_events['info']['Events'][key]['dim__']
                if len(dim_value) > 1:  # Check if dim__ exists and has multiple values
                    poke_count = dim_value[1]  # Take the second value from 'dim__'
                    total_pokes += poke_count

                    # Track poke identities
                    if key in poke_identities:
                        poke_identities[key] += poke_count
                else:
                    print("Error: No second value exists")
        poke_counts.append(total_pokes)

    return poke_counts, poke_identities
# Plot poke distributions and identities for hit and non-hit trials
for subject in subjects:
    result = dataframes_trials[subject]

    # For hit trials
    hit_poke_counts, hit_poke_identities = process_parsed_events_all_trials(zip(result['parsed_events'], result['hit']),
                                                                            hit_only=True)

    # For non-hit trials
    non_hit_poke_counts, non_hit_poke_identities = process_parsed_events_all_trials(
        zip(result['parsed_events'], result['hit']), hit_only=False)

    #Create plot of poke distribution hit trials
    plt.figure(figsize=(6, 4))
    hit_counts = pd.Series(hit_poke_counts).value_counts().sort_index()
    hit_counts.plot(kind='bar', color=color_palette[subject]['hit'], edgecolor='black')
    plt.title(f'Hit Trials - Poke Distribution\n{subject}')
    plt.xlabel('Number of Pokes')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    #plot poke distribution for non-hit trials
    plt.figure(figsize=(6, 4))
    non_hit_counts = pd.Series(non_hit_poke_counts).value_counts().sort_index()
    non_hit_counts.plot(kind='bar', color=color_palette[subject]['non_hit'], edgecolor='black')
    plt.title(f'Non-Hit Trials - Poke Distribution\n{subject}')
    plt.xlabel('Number of Pokes')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Plot poke identities for hit trials
    plt.figure(figsize=(8, 5))
    poke_keys = list(hit_poke_identities.keys())
    poke_values = list(hit_poke_identities.values())
    plt.bar(poke_keys, poke_values, color=color_palette[subject]['hit'], edgecolor='black', alpha=0.7)
    plt.title(f'Hit Trials - Poke Identities\n{subject}')
    plt.xlabel('Poke Identity')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Plot poke identities for non-hit trials
    plt.figure(figsize=(8, 5))
    poke_keys = list(non_hit_poke_identities.keys())
    poke_values = list(non_hit_poke_identities.values())
    plt.bar(poke_keys, poke_values, color=color_palette[subject]['non_hit'], edgecolor='black', alpha=0.7)
    plt.title(f'Non-Hit Trials - Poke Identities\n{subject}')
    plt.xlabel('Poke Identity')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Calculate trial type performance
for subject in subjects:
    result = dataframes_trials[subject]

    # Count left and right trials
    left_trials = result[result['parsed_events'].apply(lambda x: json.loads(x)['vals'].get('SoundCueSide') == 'L')]
    right_trials = result[result['parsed_events'].apply(lambda x: json.loads(x)['vals'].get('SoundCueSide') == 'R')]


    # Performance calculation
    def calculate_performance(trials):
        total_trials = len(trials)
        hits = trials[trials['hit'] == 1]

        # Extract SoundCueSide from the 'data' column JSON
        correct_choice = trials[trials.apply(lambda row:
                                             json.loads(row['data'])['vals'].get('choice') ==
                                             json.loads(row['data'])['vals'].get('SoundCueSide'), axis=1)]

        return {
            'total_trials': total_trials,
            'hit_percentage': len(hits) / total_trials * 100,
            'correct_choice_percentage': len(correct_choice) / total_trials * 100
        }


    # Calculate trial type performance
    for subject in subjects:
        result = dataframes_trials[subject]

        # Count left and right trials using the 'data' column
        left_trials = result[result['data'].apply(lambda x: json.loads(x)['vals'].get('SoundCueSide') == 'L')]
        right_trials = result[result['data'].apply(lambda x: json.loads(x)['vals'].get('SoundCueSide') == 'R')]

        left_performance = calculate_performance(left_trials)
        right_performance = calculate_performance(right_trials)

        print(f"\nSubject {subject} Performance:")
        print("Left Trials:")
        print(f"  Total Trials: {left_performance['total_trials']}")
        print(f"  Hit Percentage: {left_performance['hit_percentage']:.2f}%")
        print(f"  Correct Choice Percentage: {left_performance['correct_choice_percentage']:.2f}%")
        print("Right Trials:")
        print(f"  Total Trials: {right_performance['total_trials']}")
        print(f"  Hit Percentage: {right_performance['hit_percentage']:.2f}%")
        print(f"  Correct Choice Percentage: {right_performance['correct_choice_percentage']:.2f}%")


def calculate_session_bias(trials):
    # Group trials by session ID
    session_groups = trials.groupby('sessid')
    biases = []

    for _, session_data in session_groups:
        # Calculate L hit % and R hit %
        left_trials = session_data[
            session_data['data'].apply(lambda x: json.loads(x)['vals'].get('SoundCueSide') == 'L')]
        right_trials = session_data[
            session_data['data'].apply(lambda x: json.loads(x)['vals'].get('SoundCueSide') == 'R')]

        left_hits = len(left_trials[left_trials['hit'] == 1])
        right_hits = len(right_trials[right_trials['hit'] == 1])

        left_hit_percentage = (left_hits / len(left_trials)) * 100 if len(left_trials) > 0 else 0
        right_hit_percentage = (right_hits / len(right_trials)) * 100 if len(right_trials) > 0 else 0

        # Calculate bias
        if left_hit_percentage + right_hit_percentage > 0:
            bias = (right_hit_percentage - left_hit_percentage) / (right_hit_percentage + left_hit_percentage)
        else:
            bias = 0  # No trials for this session

        biases.append(bias)

    return biases


def plot_session_bias(dataframes_trials, color_palette, line_styles):
    # Find the maximum number of sessions across all subjects
    max_sessions = max(len(calculate_session_bias(dataframes_trials[subject])) for subject in subjects)

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot bias for each subject
    for subject in subjects:
        result = dataframes_trials[subject]

        # Calculate session-wise bias
        biases = calculate_session_bias(result)

        # Create evenly spaced x-axis points
        x_points = list(range(1, len(biases) + 1))

        # Use subject-specific hit color and line style
        subject_color = color_palette[subject]['hit']

        plt.plot(
            x_points,
            biases,
            marker='o',
            color=subject_color,
            linestyle=line_styles[subject],
            label=subject
        )

    # Styling
    plt.xlabel('Session', fontsize=14)
    plt.ylabel('Bias', fontsize=14)
    plt.ylim(-1.1, 1.1)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Add horizontal line at y=0

    plt.legend(title='Subject', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

plot_session_bias(dataframes_trials, color_palette, line_styles)

# Calculation of Percentage of non-hit trials with 0 pokes
def calculate_zero_poke_trials(trials):
    non_hit_trials = trials[trials['hit'] == 0]

    zero_poke_trials = 0
    total_non_hit_trials = len(non_hit_trials)

    for trial, is_hit in zip(non_hit_trials['parsed_events'], non_hit_trials['hit']):
        parsed_events = json.loads(trial)

        # Count pokes
        poke_events = {key: value for key, value in parsed_events['vals']['Events'].items() if "In" in key}

        total_pokes = 0
        for key, value in poke_events.items():
            if key in parsed_events['info']['Events'] and "In" in key:
                dim_value = parsed_events['info']['Events'][key]['dim__']
                if len(dim_value) > 1:
                    total_pokes += dim_value[1]

        if total_pokes == 0:
            zero_poke_trials += 1

    # Calculate percentage
    zero_poke_percentage = (zero_poke_trials / total_non_hit_trials) * 100 if total_non_hit_trials > 0 else 0

    return zero_poke_percentage, zero_poke_trials, total_non_hit_trials


# Calculation of RT and plotting of RT per session
def analyze_reaction_times(dataframes_trials, dataframes_sess, color_palette, line_styles):
    # Overall average RT for each subject
    for subject in subjects:
        trials = dataframes_trials[subject]
        overall_avg_rt = trials['RT'].mean()
        print(f"Subject {subject} - Overall Average Reaction Time: {overall_avg_rt:.4f}")

    # Plot Average RT per Session
    plt.figure(figsize=(12, 6))

    # Find the maximum number of sessions across all subjects
    max_sessions = max(len(dataframes_sess[subject]) for subject in subjects)

    for subject in subjects:
        # Group trials by session and calculate mean RT for each session
        trials = dataframes_trials[subject]

        rt_per_session = trials.groupby('sessid')['RT'].mean()

        # Create x points from 1 to max_sessions
        x_points = list(range(1, len(rt_per_session) + 1))

        # Use subject-specific hit color for plotting
        subject_color = color_palette[subject]['hit']

        # Plot RT for each session
        plt.plot(
            x_points,  # Use the actual number of existing sessions
            rt_per_session,
            marker='o',
            color=subject_color,
            label=subject,
            linestyle = line_styles[subject]
        )

    plt.xlabel('Session', fontsize = 14)
    plt.ylabel('Average Reaction Time', fontsize = 14)
    #plt.title('Average Reaction Time Across Sessions', fontsize = 12)
    plt.legend(title='Subject', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

#Calculate average RT and plot reaction time per session
analyze_reaction_times(dataframes_trials, dataframes_sess, color_palette, line_styles)

# Calculations of % of 0 pokes in not initiated trials
print("Zero Poke Trials Analysis:")
for subject in subjects:
    result = dataframes_trials[subject]
    zero_poke_percentage, zero_poke_trials, total_non_hit_trials = calculate_zero_poke_trials(result)

    print(f"\nSubject {subject}:")
    print(f"  Total Non-Hit Trials: {total_non_hit_trials}")
    print(f"  Zero Poke Trials: {zero_poke_trials}")
    print(f"  Percentage of Zero Poke Trials: {zero_poke_percentage:.2f}%")


def calculate_zero_poke_trials_per_session(dataframes_trials):
    zero_poke_percentages = {}

    for subject in subjects:
        trials = dataframes_trials[subject]

        # Group trials by session
        session_groups = trials.groupby('sessid')

        zero_poke_percentages[subject] = []

        for sessid, session_data in session_groups:
            # Total trials in the session
            total_trials = len(session_data)

            # Count zero-poke trials
            zero_poke_trials = 0
            for trial in session_data['parsed_events']:
                parsed_events = json.loads(trial)

                # Count pokes
                poke_events = {key: value for key, value in parsed_events['vals']['Events'].items() if "In" in key}

                total_pokes = 0
                for key, value in poke_events.items():
                    if key in parsed_events['info']['Events'] and "In" in key:
                        dim_value = parsed_events['info']['Events'][key]['dim__']
                        if len(dim_value) > 1:
                            total_pokes += dim_value[1]

                if total_pokes == 0:
                    zero_poke_trials += 1

            # Calculate percentage
            zero_poke_percentage = (zero_poke_trials / total_trials) * 100

            zero_poke_percentages[subject].append(zero_poke_percentage)

    return zero_poke_percentages


def plot_zero_poke_trials_per_session(zero_poke_percentages, color_palette, line_styles):
    plt.figure(figsize=(12, 6))

    for subject in subjects:
        # Create x points from 1 to number of sessions
        x_points = list(range(1, len(zero_poke_percentages[subject]) + 1))

        # Use subject-specific hit color for plotting
        subject_color = color_palette[subject]['hit']

        plt.plot(
            x_points,
            zero_poke_percentages[subject],
            marker='o',
            color=subject_color,
            label=subject,
            linestyle=line_styles[subject]
        )

    plt.xlabel('Session', fontsize=14)
    plt.ylabel('Zero-Poke Trials (%)', fontsize=14)
    plt.legend(title='Subject', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# Calculate zero-poke trials percentages
zero_poke_percentages = calculate_zero_poke_trials_per_session(dataframes_trials)

# Plot zero-poke trials percentages
plot_zero_poke_trials_per_session(zero_poke_percentages, color_palette, line_styles)