import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Datenbankpfad anpassen (z.B. im Download-Ordner)
db_path = r"C:\Users\elias\Downloads\markt_scraper_analyse\monitoring.db"


# --- Read Data from Database ---
conn = sqlite3.connect(db_path)
df_monitor = pd.read_sql_query("SELECT * FROM monitor_records", conn)
df_profiles = pd.read_sql_query("SELECT * FROM profiles", conn)
conn.close()

# --- Merge Tables ---
# We merge on ad_unique_id and push_counter (each push is treated as a separate ad)
df = pd.merge(
    df_monitor, 
    df_profiles[['ad_unique_id', 'push_counter', 'age', 'profile_url']], 
    on=['ad_unique_id', 'push_counter'], 
    how='inner'
)

# --- Filter and Process Age ---
# Remove rows with missing or empty age, and convert age to numeric
df = df[df['age'].notnull() & (df['age'] != '')].copy()
df['age_numeric'] = pd.to_numeric(df['age'], errors='coerce')
df = df[df['age_numeric'].notnull()]

# Assign age groups:
# Group 1: 18-19, Group 2: 20-25, Group 3: 26+
def assign_age_group(age):
    if 18 <= age <= 19:
        return "18-19"
    elif 20 <= age <= 25:
        return "20-25"
    elif age >= 26:
        return "26+"
    else:
        return None

df['age_group'] = df['age_numeric'].apply(assign_age_group)
df = df[df['age_group'].notnull()].copy()

# --- Categorize Based on profile_url ---
# Check if profile_url contains specific substrings to assign the category
df['category'] = np.where(
    df['profile_url'].str.contains("sexbilder-sexvideos", case=False, na=False),
    "sexbilder-sexvideos",
    np.where(
        df['profile_url'].str.contains("sexchat", case=False, na=False),
        "sexchat",
        "other"
    )
)
df = df[df['category'].isin(["sexbilder-sexvideos", "sexchat"])].copy()

# --- Helper Functions ---
def compute_view_rate(df_sub):
    """
    For each ad push (defined by ad_unique_id and push_counter), compute the view rate.
    The rate is computed as the difference in views (delta_views) over the time interval (delta_time) 
    converted to views per hour. Negative rates (due to measurement noise) are clipped to 0.
    """
    df_sub['ad_push'] = df_sub['ad_unique_id'] + "_" + df_sub['push_counter'].astype(str)
    df_sub = df_sub.sort_values(by=["ad_push", "ad_age_in_minutes"])
    df_sub['delta_views'] = df_sub.groupby('ad_push')['views'].diff()
    df_sub['delta_time'] = df_sub.groupby('ad_push')['ad_age_in_minutes'].diff()
    df_sub = df_sub.dropna(subset=['delta_views', 'delta_time'])
    df_sub = df_sub[df_sub['delta_time'] > 0]
    df_sub['rate'] = df_sub['delta_views'] * 60 / df_sub['delta_time']  # views per hour
    df_sub['rate'] = df_sub['rate'].clip(lower=0)  # Remove accidental negatives
    df_sub['mid_age'] = df_sub['ad_age_in_minutes'] - df_sub['delta_time'] / 2
    return df_sub

def bin_and_smooth(df_sub, bin_size=5, smoothing_window=5):
    """
    Bins the view rate data into intervals of bin_size (in minutes) from 0 to 1440 (24h).
    Then applies a rolling average with the specified smoothing_window to reduce extreme values.
    """
    bins = np.arange(0, 1440 + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    df_sub['bin'] = pd.cut(df_sub['mid_age'], bins=bins, labels=bin_centers, include_lowest=True)
    binned = df_sub.groupby('bin')['rate'].mean().reset_index()
    binned['bin'] = binned['bin'].astype(float)
    binned = binned.sort_values(by='bin')
    # Apply rolling average smoothing
    binned['smoothed_rate'] = binned['rate'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    binned['time_hours'] = binned['bin'] / 60.0  # convert minutes to hours for plotting
    return binned

# --- Process Data for Each (Category, Age Group) Combination ---
results = {}
categories = ["sexbilder-sexvideos", "sexchat"]
age_groups = ["18-19", "20-25", "26+"]

for cat in categories:
    for age_grp in age_groups:
        key = (cat, age_grp)
        df_subset = df[(df['category'] == cat) & (df['age_group'] == age_grp)].copy()
        if df_subset.empty:
            continue
        df_subset = compute_view_rate(df_subset)
        binned = bin_and_smooth(df_subset, bin_size=5, smoothing_window=5)
        results[key] = binned

# --- Plotting ---
# Two plots (one per category) with 3 curves each (one per age group)
colors = {
    "18-19": "blue",
    "20-25": "green",
    "26+": "red"
}

fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for i, cat in enumerate(categories):
    ax = axs[i]
    for age_grp in age_groups:
        key = (cat, age_grp)
        if key in results:
            binned = results[key]
            ax.plot(binned['time_hours'], binned['smoothed_rate'], marker='o', linestyle='-', 
                    color=colors[age_grp], label=f"Age {age_grp}")
    ax.set_title(f"Category: {cat}")
    ax.set_xlabel("Time since posting (hours)")
    ax.set_xlim(0, 24)
    ax.set_ylabel("Views Rate (views per hour)")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()






import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math # For ceiling function

# --- Configuration ---
MAX_AGE_MINUTES = 119 # Consider records up to this age for initial rate calculation
RATE_SMOOTHING_WINDOW = 3 # Window size for smoothing the rate line plot (e.g., 3 hours)
COUNT_BIN_MINUTES = 10 # Interval size for counting new ad pushes

# --- Read Data from Database ---
try:
    conn = sqlite3.connect(db_path)
    # Read necessary columns for rate calculation
    df_monitor = pd.read_sql_query(
        "SELECT id, ad_unique_id, push_counter, track_time, ad_age_in_minutes, views FROM monitor_records", conn
    )
    # Read necessary columns for push counting and merging
    df_profiles = pd.read_sql_query(
        "SELECT ad_unique_id, push_counter, posting_time FROM profiles", conn
    )
    conn.close()
except Exception as e:
    print(f"Error connecting to or reading from database: {e}")
    exit()

print(f"Loaded {len(df_monitor)} monitor records.")
print(f"Loaded {len(df_profiles)} profile records.")

# --- Data Preparation ---
# Convert time columns to datetime objects
df_monitor['track_time'] = pd.to_datetime(df_monitor['track_time'], errors='coerce')
df_profiles['posting_time'] = pd.to_datetime(df_profiles['posting_time'], errors='coerce')

# Drop rows where conversion failed or essential data is missing
df_monitor.dropna(subset=['track_time', 'ad_age_in_minutes', 'views', 'ad_unique_id', 'push_counter'], inplace=True)
df_profiles.dropna(subset=['posting_time', 'ad_unique_id', 'push_counter'], inplace=True)

# Ensure numeric types for calculations
df_monitor['ad_age_in_minutes'] = pd.to_numeric(df_monitor['ad_age_in_minutes'], errors='coerce')
df_monitor['views'] = pd.to_numeric(df_monitor['views'], errors='coerce')
df_monitor.dropna(subset=['ad_age_in_minutes', 'views'], inplace=True) # Drop if conversion failed

print(f"Profile records after cleaning: {len(df_profiles)}")
print(f"Monitor records after cleaning: {len(df_monitor)}")


# --- Step 1: Count Unique Ad Pushes Directly from Profiles ---
print("\n--- Processing Ad Push Counts ---")
# Identify unique pushes based on profile data
df_unique_pushes = df_profiles[['ad_unique_id', 'push_counter', 'posting_time']].drop_duplicates(subset=['ad_unique_id', 'push_counter'])
print(f"Found {len(df_unique_pushes)} unique ad pushes in profiles table.")

if df_unique_pushes.empty:
     print("No unique pushes identified in profiles table for counting.")
     # Decide if you want to exit or continue to rate calculation
     # exit() # Optional: exit if no pushes found
     binned_counts = pd.Series(dtype=int) # Create empty series if continuing
else:
    # Calculate minute of the day (0 to 1439)
    df_unique_pushes['posting_minute_of_day'] = df_unique_pushes['posting_time'].dt.hour * 60 + df_unique_pushes['posting_time'].dt.minute

    # Create bins for every COUNT_BIN_MINUTES minutes
    num_bins = math.ceil(24 * 60 / COUNT_BIN_MINUTES)
    bins = np.linspace(0, num_bins * COUNT_BIN_MINUTES, num_bins + 1) # e.g., [0, 10, 20, ..., 1440]

    # Create labels for the bins (e.g., "00:00", "00:10", ...)
    bin_labels = [f"{int(b // 60):02d}:{int(b % 60):02d}" for b in bins[:-1]]

    # Assign each push to a time bin
    df_unique_pushes['time_bin'] = pd.cut(df_unique_pushes['posting_minute_of_day'],
                                   bins=bins,
                                   labels=bin_labels,
                                   right=False, # Intervals are [start, end)
                                   include_lowest=True)

    # Count pushes per bin
    binned_counts = df_unique_pushes.groupby('time_bin', observed=False).size() # Use observed=False with CategoricalIndex

    # Ensure all bins are present, fill missing with 0
    binned_counts = binned_counts.reindex(bin_labels, fill_value=0)

    print(f"\nAd Push Counts per {COUNT_BIN_MINUTES}-minute Interval (from Profiles):")
    print(binned_counts[binned_counts > 0]) # Print only non-zero counts for brevity


# --- Step 2: Calculate Initial Views per Hour (Requires Merging) ---
print("\n--- Processing Initial View Rates ---")
# Merge monitor data with profile data (posting_time is needed for grouping by hour later)
# Use the original df_profiles which might have duplicate pushes if scraped multiple times,
# but we only need one posting_time per push, so drop duplicates before merge.
df_merged_for_rates = pd.merge(
    df_monitor,
    df_profiles[['ad_unique_id', 'push_counter', 'posting_time']].drop_duplicates(subset=['ad_unique_id', 'push_counter']),
    on=['ad_unique_id', 'push_counter'],
    how='inner' # Only keep monitor records that have a corresponding profile entry
)

print(f"Merged dataframe for rate calculation has {len(df_merged_for_rates)} rows.")

# Filter for records within the initial time window
df_initial = df_merged_for_rates[df_merged_for_rates['ad_age_in_minutes'] <= MAX_AGE_MINUTES].copy()
print(f"Filtered down to {len(df_initial)} records within the first {MAX_AGE_MINUTES} minutes for rate calc.")

# Group by each specific ad push to calculate rates
grouped = df_initial.groupby(['ad_unique_id', 'push_counter'])

rate_results = []
processed_pushes_for_rate = 0

for name, group in grouped:
    # Ensure group is sorted by age to easily find first and last record
    group = group.sort_values('ad_age_in_minutes')

    # Need at least two distinct time points to calculate a rate
    if len(group) >= 2 and group['ad_age_in_minutes'].nunique() > 1:
        first_record = group.iloc[0]
        last_record = group.iloc[-1] # Last record within the MAX_AGE_MINUTES window

        delta_views = last_record['views'] - first_record['views']
        delta_time_minutes = last_record['ad_age_in_minutes'] - first_record['ad_age_in_minutes']

        # Avoid division by zero
        if delta_time_minutes > 0:
            views_per_hour = (delta_views / delta_time_minutes) * 60
            # Clip negative rates
            views_per_hour = max(0, views_per_hour)

            # Get the posting hour (should be the same for all records in the group)
            posting_hour = first_record['posting_time'].hour

            rate_results.append({
                'ad_unique_id': name[0],
                'push_counter': name[1],
                'posting_hour': posting_hour,
                'initial_views_per_hour': views_per_hour
            })
            processed_pushes_for_rate += 1

if not rate_results:
    print("No ad pushes found with sufficient data points in the initial period to calculate any rates.")
    # Create empty dataframe and series to avoid errors later
    df_rates = pd.DataFrame(columns=['posting_hour', 'initial_views_per_hour'])
    smoothed_hourly_avg_rates = pd.Series(index=range(24), data=0.0)
else:
    df_rates = pd.DataFrame(rate_results)
    print(f"Calculated initial rates for {processed_pushes_for_rate} ad pushes.")

    # --- Aggregate Rate by Posting Hour and Smooth ---
    hourly_avg_rates = df_rates.groupby('posting_hour')['initial_views_per_hour'].mean()
    # Ensure all hours 0-23 are present
    hourly_avg_rates = hourly_avg_rates.reindex(range(24), fill_value=0)
    # Apply smoothing
    smoothed_hourly_avg_rates = hourly_avg_rates.rolling(window=RATE_SMOOTHING_WINDOW, center=True, min_periods=1).mean()

    print("\nSmoothed Average Initial Views per Hour by Posting Hour:")
    print(smoothed_hourly_avg_rates)


# --- Step 3: Plotting ---
print("\n--- Generating Plot ---")
fig, ax1 = plt.subplots(figsize=(18, 7)) # Wider figure

# Plot 1: Smoothed Average View Rate (Line Plot)
color1 = 'tab:blue'
ax1.set_xlabel('Hour of Day Posted')
ax1.set_ylabel('Avg. Initial Views/Hour (Smoothed)', color=color1)
# Use range(24) for x-coordinates matching the hourly aggregation
ax1.plot(smoothed_hourly_avg_rates.index, smoothed_hourly_avg_rates.values, color=color1, marker='o', linestyle='-', label='Avg. Views/Hour')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, axis='y', linestyle='--')
ax1.set_xticks(range(24)) # Ensure ticks for every hour
ax1.set_xlim(-0.5, 23.5) # Add slight padding
# Set y-limit for rate, starting from 0
ax1.set_ylim(bottom=0)

# Plot 2: Ad Push Count (Bar Plot on Secondary Axis)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color2 = 'tab:red'
ax2.set_ylabel(f'New Ad Pushes per {COUNT_BIN_MINUTES} min', color=color2)

# Check if binned_counts is not empty before plotting
if not binned_counts.empty:
    # Calculate bar positions - they should align within the hours
    bar_width = 0.8 * (COUNT_BIN_MINUTES / 60.0) # Make bars slightly narrower than the interval fraction of an hour
    bar_positions = np.arange(len(binned_counts)) * (COUNT_BIN_MINUTES / 60.0) # Position based on fraction of hour

    ax2.bar(bar_positions, binned_counts.values, color=color2, alpha=0.6, width=bar_width, label=f'Ad Pushes/{COUNT_BIN_MINUTES}min')
else:
    print("Skipping plotting ad push counts as no data was available.")


ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(bottom=0) # Ensure y-axis starts at 0

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
# Check if ax2 has plotted anything before getting its legend
if not binned_counts.empty:
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
else:
     ax1.legend(loc='upper left')


plt.title(f'Initial Ad Performance and Posting Volume vs. Time of Day (Window: {MAX_AGE_MINUTES} min)')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()




import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
MIN_ADS_PER_CITY = 10 # Minimum number of ads per city to include in analysis
INITIAL_MAX_AGE_MINUTES = 40 # Ad must have a record at or before this age
ANALYSIS_MAX_AGE_MINUTES = 400 # 6 hours * 60 min/hour

# --- Read Data from Database ---
try:
    conn = sqlite3.connect(db_path)
    # Select necessary columns
    df_monitor = pd.read_sql_query(
        "SELECT ad_unique_id, push_counter, ad_age_in_minutes, views FROM monitor_records", conn
    )
    df_profiles = pd.read_sql_query(
        "SELECT ad_unique_id, push_counter, city FROM profiles", conn
    )
    conn.close()
except Exception as e:
    print(f"Error connecting to or reading from database: {e}")
    exit()

print(f"Loaded {len(df_monitor)} monitor records.")
print(f"Loaded {len(df_profiles)} profile records.")

# --- Data Preparation ---
# Convert data types and handle missing values
df_monitor['ad_age_in_minutes'] = pd.to_numeric(df_monitor['ad_age_in_minutes'], errors='coerce')
df_monitor['views'] = pd.to_numeric(df_monitor['views'], errors='coerce')
df_monitor.dropna(subset=['ad_unique_id', 'push_counter', 'ad_age_in_minutes', 'views'], inplace=True)

df_profiles['city'] = df_profiles['city'].str.strip() # Remove leading/trailing whitespace
df_profiles.replace('', np.nan, inplace=True) # Replace empty strings with NaN
df_profiles.dropna(subset=['ad_unique_id', 'push_counter', 'city'], inplace=True)

# Ensure keys are strings for consistent merging if needed, though usually not necessary if types match
# df_monitor['ad_unique_id'] = df_monitor['ad_unique_id'].astype(str)
# df_profiles['ad_unique_id'] = df_profiles['ad_unique_id'].astype(str)
# df_monitor['push_counter'] = df_monitor['push_counter'].astype(str) # Assuming push_counter is numeric in DB
# df_profiles['push_counter'] = df_profiles['push_counter'].astype(str)

print(f"Monitor records after cleaning: {len(df_monitor)}")
print(f"Profile records after cleaning: {len(df_profiles)}")


# --- Merge Tables ---
# Use drop_duplicates on profiles before merge to ensure one city per ad push
df_merged = pd.merge(
    df_monitor,
    df_profiles.drop_duplicates(subset=['ad_unique_id', 'push_counter']),
    on=['ad_unique_id', 'push_counter'],
    how='inner' # Keep only monitor records with a matching profile (and city)
)
print(f"Merged dataframe size: {len(df_merged)}")


# --- Filter by City Ad Count ---
print(f"\n--- Filtering Cities with less than {MIN_ADS_PER_CITY} ads ---")
# Count unique ad pushes per city using the merged data (or profiles if preferred)
# It's safer to count based on the profiles data before the merge potentially drops pushes without monitoring data
city_ad_counts = df_profiles.drop_duplicates(subset=['ad_unique_id', 'push_counter'])['city'].value_counts()

# Identify cities meeting the threshold
cities_to_keep = city_ad_counts[city_ad_counts >= MIN_ADS_PER_CITY].index.tolist()
print(f"Found {len(cities_to_keep)} cities with >= {MIN_ADS_PER_CITY} ads.")
# print("Cities being kept:", cities_to_keep) # Optional: print the list

if not cities_to_keep:
    print("No cities meet the minimum ad count requirement. Exiting.")
    exit()

# Filter the merged dataframe to keep only data from these cities
df_filtered_cities = df_merged[df_merged['city'].isin(cities_to_keep)].copy()
print(f"Dataframe size after filtering by city count: {len(df_filtered_cities)}")


# --- Calculate Views Gained per Ad Push (0-6 hours) ---
print("\n--- Calculating views gained in first 6 hours ---")
grouped = df_filtered_cities.groupby(['ad_unique_id', 'push_counter'])

results = []
processed_pushes = 0
discarded_pushes_no_initial = 0
discarded_pushes_no_final = 0
discarded_pushes_time_issue = 0

for name, group in grouped:
    processed_pushes += 1
    group = group.sort_values('ad_age_in_minutes')

    # Find earliest record within INITIAL_MAX_AGE_MINUTES
    initial_records = group[group['ad_age_in_minutes'] <= INITIAL_MAX_AGE_MINUTES]
    if initial_records.empty:
        discarded_pushes_no_initial += 1
        continue # Skip this ad push if no early record found
    earliest_record = initial_records.iloc[0] # First one after sorting

    # Find latest record within ANALYSIS_MAX_AGE_MINUTES
    final_records = group[group['ad_age_in_minutes'] <= ANALYSIS_MAX_AGE_MINUTES]
    if final_records.empty:
        # This case is unlikely if earliest_record exists, but check anyway
        discarded_pushes_no_final += 1
        continue
    latest_record = final_records.iloc[-1] # Last one after sorting

    # Ensure we have distinct points in time and the latest is actually later
    if latest_record['ad_age_in_minutes'] <= earliest_record['ad_age_in_minutes']:
        discarded_pushes_time_issue +=1
        continue # Skip if latest isn't strictly later than earliest

    # Calculate views gained
    views_gained = latest_record['views'] - earliest_record['views']
    views_gained = max(0, views_gained) # Clip at 0

    results.append({
        'city': earliest_record['city'], # City is the same for the whole group
        'ad_unique_id': name[0],
        'push_counter': name[1],
        'views_gained_6hr': views_gained
    })

print(f"Processed {processed_pushes} unique ad pushes from selected cities.")
print(f"Discarded {discarded_pushes_no_initial} pushes: No record <= {INITIAL_MAX_AGE_MINUTES} min.")
print(f"Discarded {discarded_pushes_no_final} pushes: No record <= {ANALYSIS_MAX_AGE_MINUTES} min (after initial check).")
print(f"Discarded {discarded_pushes_time_issue} pushes: Latest record not later than earliest record.")

if not results:
    print("No ad pushes had valid data points for the 6-hour analysis. Cannot proceed.")
    exit()

df_views_gained = pd.DataFrame(results)
print(f"Successfully calculated 6hr view gain for {len(df_views_gained)} ad pushes.")

# --- Aggregate by City ---
city_avg_views = df_views_gained.groupby('city')['views_gained_6hr'].mean()

# Sort cities by average views gained (descending)
city_avg_views_sorted = city_avg_views.sort_values(ascending=False)

print("\n--- Average Views Gained in First 6 Hours by City ---")
print(city_avg_views_sorted)

# --- Visualize ---
print("\n--- Generating Plot ---")
plt.figure(figsize=(12, max(6, len(city_avg_views_sorted) * 0.4))) # Adjust height based on number of cities

city_avg_views_sorted.plot(kind='barh', color='teal') # Horizontal bar chart is good for many categories

plt.xlabel('Average Views Gained in First 6 Hours')
plt.ylabel('City')
plt.title(f'Ad Performance by City (Min {MIN_ADS_PER_CITY} Ads/City)')
plt.gca().invert_yaxis() # Display city with highest views at the top
plt.grid(axis='x', linestyle='--')
plt.tight_layout()
plt.show()