from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.io import show, reset_output, output_file
from bokeh.models import Range1d
from bokeh.plotting import figure
# Plot a predicted vs. true value in a square-shaped scatter plot. They should ideally lie on a diagonal
from pytz import timezone

from constants import project_mapping
from pixelgrams import normalise_into_range


def plot_prediction_scatter_plot(y_pred, y_test, max_val=40):
    sort_idxs = np.argsort(y_test)
    y_test_sorted = y_test[sort_idxs]
    y_pred_sorted = y_pred[sort_idxs]

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)

    ax.plot([0, max_val], [0, max_val], color="lightgrey")
    ax.scatter(y_test_sorted, y_pred_sorted, s=2)

    ax.set_ylim(0, max_val)
    ax.set_xlim(0, max_val)
    plt.show()


# I mainly use this as reference to remember how Bokeh plots are created
def plot_in_bokeh(x, y):
    reset_output()
    output_file("pathname")

    fig = figure(height=300, width=900, responsive=True, x_axis_type="datetime")

    fig.scatter(x, y, legend="Test")

    fig.y_range = Range1d(10, 40)
    fig.yaxis.axis_label = "BrPM"

    '''
    # Setting the second y axis range name and range
    fig.extra_y_ranges = {"act_level": Range1d(start=0, end=2)}
    # Adding the second axis to the plot.
    fig.add_layout(LinearAxis(y_range_name="act_level",axis_label="Activity level"), 'right')
    
    fig.line(act_level_x, act_level_y, y_range_name="act_level", color="lightgrey", legend="activity level")
    '''
    show(fig)


def plot_airspeck_overview_heatmap(airspeck_list, labels_list, timeframe, project_name, plot_outpath, title,
                                   column_name, all_black=False):
    sns.reset_orig()

    tz = timezone(project_mapping[project_name][1])

    if timeframe[0].tzinfo is None:
        start_date = tz.localize(timeframe[0])
        end_date = tz.localize(timeframe[1])
    else:
        start_date = timeframe[0]
        end_date = timeframe[1]

    # Round to full day
    start_date = start_date.replace(hour=0, minute=0, second=0)
    end_date = end_date.replace(hour=0, minute=0, second=0) + timedelta(days=1)

    def add_row_to_timegrid(time_grid, row_idx, values, col_name, start_date):
        # 10 minute average
        values_indexed = values.set_index('timestamp')
        values_10_avg = values_indexed.resample('10Min').mean()

        # Normalise col_name
        norm_col = normalise_into_range(values_10_avg[col_name], values_10_avg[col_name].min(),
                                        values_10_avg[col_name].max())

        for idx in range(len(values_10_avg)):
            idx_in_row = int((values_10_avg.index[idx] - start_date).total_seconds() / (60. * 10))
            if all_black:
                time_grid[row_idx, idx_in_row] = int(norm_col[idx] > 0)
            else:
                time_grid[row_idx, idx_in_row] = norm_col[idx]

    num_days = int((end_date - start_date) / pd.Timedelta(days=1))

    # Beginning of day has the date, the rest only the hour:minute
    time_labels = []
    for ts in pd.date_range(start=start_date, end=end_date, freq='4H', tz=project_mapping[project_name][1]):
        if ts.hour == 0:
            time_labels.append(datetime.strftime(ts.to_pydatetime(), "%d.%m.%Y"))
        else:
            time_labels.append(datetime.strftime(ts.to_pydatetime(), "%Hh"))

    # Matrix for storing pollution levels PM 2.5 in 10 minute intervals over num_days
    time_grid = np.zeros((len(airspeck_list), 6 * 24 * num_days))

    day_idxs = np.linspace(0, time_grid.shape[1], num_days + 1)

    for idx, airspeck in enumerate(airspeck_list):
        if len(airspeck) > 0:
            add_row_to_timegrid(time_grid, idx, airspeck, column_name, start_date)

    # Plot
    fig, ax = plt.subplots(1)
    plot_height = len(airspeck_list) / 4. + 0.5
    fig.set_size_inches((3 * num_days, plot_height))

    ax.pcolormesh(time_grid, cmap='Greys', vmin=0, vmax=1)

    # Dates x-axis
    ax.xaxis.set_ticklabels(time_labels)
    ax.set_xticks(np.arange(0, time_grid.shape[1] + 1, 6 * 4))

    for idx, t in enumerate(ax.get_xticklabels()):
        if idx % 6 == 0:
            t.set_y(-0.3 / plot_height)
        else:
            t.set_y(0)

    # Vertical lines at beginning of day
    for day in day_idxs:
        ax.vlines(day, -0.5, len(time_grid), linewidth=1.0)

    # Horizontal lines every 5 lines for better readibility
    for row_idx in range(0, time_grid.shape[0], 5):
        ax.hlines(row_idx, 0, time_grid.shape[1], linewidth=1.0)

    # Labels y axis
    ax.set_yticks(np.arange(time_grid.shape[0]) + 0.5)
    ax.yaxis.set_ticklabels(labels_list)

    ax.set_ylim(0, len(time_grid))
    ax.invert_yaxis()
    ax.set_title(title + " (0: white - Max of each sensor: black)")
    plt.tick_params(left=False)

    plt.tight_layout()
    plt.savefig(plot_outpath, dpi=300)
    plt.show()
