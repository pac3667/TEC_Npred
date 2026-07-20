import os

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

import matplotlib
try:
    matplotlib.use('module://backend_interagg')
except ModuleNotFoundError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['text.antialiased'] = True
plt.rcParams['lines.antialiased'] = True
from matplotlib import pyplot as plt, ticker

def print_stat_results(reports_dir, results, y_test, calc_goal):
    os.makedirs(reports_dir, exist_ok=True)
    print("\n" + "=" * 50)
    print("FINAL MODEL COMPARISON REPORT")
    print("=" * 50)

    report = []
    min_len = min([len(p) for p in results.values()])
    for name, pred in results.items():
        y_pred_final = pred[-min_len:]
        y_true_final = y_test[-min_len:]

        mae = metrics.mean_absolute_error(y_true_final, y_pred_final)
        rmse = np.sqrt(metrics.mean_squared_error(y_true_final, y_pred_final))
        wape  = np.sum(np.abs(y_true_final - y_pred_final)) / np.sum(np.abs(y_true_final)) * 100
        r2 = metrics.r2_score(y_true_final, y_pred_final)

        report.append({
            'Model Name': name,
            'MAE': mae,
            'RMSE': rmse,
            'WAPE (%)': wape,
            'R2 Score': r2
        })

    df_report = pd.DataFrame(report)
    df_report = df_report.sort_values(by='MAE').reset_index(drop=True)

    print(df_report.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    if calc_goal == 'TEC':
        df_report.to_csv(reports_dir + 'Power_Station_model_evaluation_report.csv', index=False)
    else:
        df_report.to_csv(reports_dir + calc_goal+'_model_evaluation_report.csv', index=False)
    print("\n[INFO] Report saved to 'model_evaluation_report.csv'")

    min_len = min([len(p) for p in results.values()])
    export_df = pd.DataFrame({'Actual_N': y_test[-min_len:].flatten()})

    for name, pred in results.items():
        export_df[name] = pred[-min_len:].flatten()

    if calc_goal == 'TEC':
        export_df.to_excel(reports_dir + 'Power_Station_final_predictions_comparison.xlsx', index=False)
    else:
        export_df.to_excel(reports_dir + calc_goal+'_final_predictions_comparison.xlsx', index=False)
    print("[INFO] Predictions exported to 'final_predictions_comparison.xlsx'")

    window_models = df_report[df_report['Model Name'].str.endswith('_window')]

    best_window_model_name = ''

    if not window_models.empty:
        best_model = window_models.iloc[0]

        best_window_model_name = best_model['Model Name']


    standard_models = df_report[~df_report['Model Name'].str.endswith('_window')]
    best_stat_model_name = ''

    if not standard_models.empty:
        best_model = standard_models.iloc[0]

        best_stat_model_name = best_model['Model Name']

    return best_window_model_name, best_stat_model_name
def save_calibration_predictions(reports_dir, calibration_results, y_calibration, calc_goal):
    if not calibration_results:
        return
    os.makedirs(reports_dir, exist_ok=True)
    min_len = min([len(p) for p in calibration_results.values()])
    export_df = pd.DataFrame({'Actual_N': y_calibration[-min_len:].flatten()})

    for name, pred in calibration_results.items():
        export_df[name] = pred[-min_len:].flatten()

    if calc_goal == 'TEC':
        filepath = reports_dir + 'Power_Station_calibration_predictions_comparison.csv'
    else:
        filepath = reports_dir + calc_goal + '_calibration_predictions_comparison.csv'
    export_df.to_csv(filepath, index=False)
    print(f"[INFO] Calibration predictions exported to '{filepath}'")
def print_step_results(lstm_multi_results, cbr_multi_results):
    best_step_model = ''
    print("\n" + "=" * 60)
    print("HORIZON ANALYSIS (LSTM vs CBR)")
    print("-" * 60)
    print(f"{'Day':<5} | {'LSTM MAE':<12} | {'LSTM MSE':<12} | {'LSTM WAPE':<12} | {'LSTM R2':<12} | {'CBR MAE':<12} | {'CBR MSE':<12} | {'CBR WAPE':<12} | {'CBR R2':<12}")

    for i in range(lstm_multi_results.shape[0]):
        print(f"{i + 1:<5} | {lstm_multi_results[i][0]:<12.4f} | {lstm_multi_results[i][1]:<12.4f} | {lstm_multi_results[i][2]:<12.4f} | {lstm_multi_results[i][3]:<12.4f} | {cbr_multi_results[i][0]:<12.4f} | {cbr_multi_results[i][1]:<12.4f} | {cbr_multi_results[i][2]:<12.4f} | {cbr_multi_results[i][3]:<12.4f}")

    lstm_mean_wape = np.nanmean(lstm_multi_results[:, 2])
    cbr_mean_wape = np.nanmean(cbr_multi_results[:, 2])
    lstm_mean_mae = np.nanmean(lstm_multi_results[:, 0])
    cbr_mean_mae = np.nanmean(cbr_multi_results[:, 0])
    print("-" * 60)
    print(f"Mean WAPE: LSTM={lstm_mean_wape:.4f}, CBR={cbr_mean_wape:.4f}")
    print(f"Mean MAE:  LSTM={lstm_mean_mae:.4f}, CBR={cbr_mean_mae:.4f}")

    if lstm_mean_wape < cbr_mean_wape:
        best_step_model = 'lstm'
    elif cbr_mean_wape < lstm_mean_wape:
        best_step_model = 'CBR'
    else:
        best_step_model = 'lstm' if lstm_mean_mae <= cbr_mean_mae else 'CBR'
    print(f"Best direct multistep model by mean WAPE: {best_step_model}")
    return best_step_model


def save_step_predictions(
    reports_dir,
    calc_goal,
    lstm_multi_results,
    cbr_multi_results,
    lstm_multi_metrics,
    cbr_multi_metrics,
    best_step_model_name,
):
    os.makedirs(reports_dir, exist_ok=True)
    horizon_count = lstm_multi_results.shape[1]
    horizon_cols = [f"h{idx:02d}" for idx in range(1, horizon_count + 1)]
    prefix = "Power_Station" if calc_goal == "TEC" else calc_goal

    pd.DataFrame(lstm_multi_results, columns=horizon_cols).to_csv(
        os.path.join(reports_dir, f"{prefix}_LSTM_direct_multistep_predictions.csv"),
        index=False,
    )
    pd.DataFrame(cbr_multi_results, columns=horizon_cols).to_csv(
        os.path.join(reports_dir, f"{prefix}_CBR_direct_multistep_predictions.csv"),
        index=False,
    )

    best_predictions = lstm_multi_results if best_step_model_name == "lstm" else cbr_multi_results
    pd.DataFrame(best_predictions, columns=horizon_cols).to_csv(
        os.path.join(reports_dir, f"{prefix}_best_direct_multistep_predictions.csv"),
        index=False,
    )

    metrics_df = pd.DataFrame({
        "horizon": horizon_cols,
        "LSTM_MAE": lstm_multi_metrics[:, 0],
        "LSTM_MSE": lstm_multi_metrics[:, 1],
        "LSTM_WAPE_percent": lstm_multi_metrics[:, 2],
        "LSTM_R2": lstm_multi_metrics[:, 3],
        "CBR_MAE": cbr_multi_metrics[:, 0],
        "CBR_MSE": cbr_multi_metrics[:, 1],
        "CBR_WAPE_percent": cbr_multi_metrics[:, 2],
        "CBR_R2": cbr_multi_metrics[:, 3],
        "best_model": best_step_model_name,
    })
    metrics_df.to_csv(
        os.path.join(reports_dir, f"{prefix}_direct_multistep_metrics.csv"),
        index=False,
    )
    print(f"[INFO] Direct multistep predictions exported for {prefix}")
def plot_compare_models_violations(y_true_real, y_pred_custom, y_pred_mae, n_max, n_min, title_suffix=""):
    plt.figure(figsize=(25, 12))

    steps = np.arange(len(y_true_real))
    y_true = y_true_real.flatten()
    y_cust = y_pred_custom.flatten()
    y_mae = y_pred_mae.flatten()
    n_max_f = n_max.flatten()
    n_min_f = n_min.flatten()

    plt.fill_between(steps, n_min_f, n_max_f, color='gray', alpha=0.06, label='Valid Zone')

    plt.plot(steps, n_max_f, '--', color='#e74c3c', alpha=0.5, linewidth=2.4, label='Upper Bound (n_max)')
    plt.plot(steps, n_min_f, '--', color='#9b59b6', alpha=0.5, linewidth=2.4, label='Lower Bound (n_min)')

    plt.plot(steps, y_true, label='True Values', color='#2ed573', alpha=0.7, linewidth=3.0)
    plt.plot(steps, y_cust, label='Predictions (Custom Loss)', color='#1e90ff', linewidth=4.0)

    cust_over = y_cust > n_max_f
    cust_under = y_cust < n_min_f
    if np.any(cust_over):
        plt.scatter(steps[cust_over], y_cust[cust_over], color='#0056b3', marker='v', s=140, zorder=5,
                    label=f'Custom: Over Max ({np.sum(cust_over)} pts)')
    if np.any(cust_under):
        plt.scatter(steps[cust_under], y_cust[cust_under], color='#0056b3', marker='^', s=140, zorder=5,
                    label=f'Custom: Under Min ({np.sum(cust_under)} pts)')

    plt.plot(steps, y_mae, label='Predictions (MAE Loss)', color='#f39c12', linewidth=3.6, linestyle='-.')

    mae_over = y_mae > n_max_f
    mae_over_under = y_mae < n_min_f
    if np.any(mae_over):
        plt.scatter(steps[mae_over], y_mae[mae_over], color='#d35400', marker='v', s=140, zorder=4,
                    label=f'MAE: Over Max ({np.sum(mae_over)} pts)')
    if np.any(mae_over_under):
        plt.scatter(steps[mae_over_under], y_mae[mae_over_under], color='#d35400', marker='^', s=140, zorder=4,
                    label=f'MAE: Under Min ({np.sum(mae_over_under)} pts)')

    plt.title(f"Comparison of Boundary Violations: Custom vs MAE {title_suffix}", fontsize=28, fontweight='bold', pad=30)
    plt.xlabel("Time Steps / Samples", fontsize=24, labelpad=15)
    plt.ylabel("Value", fontsize=24, labelpad=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, shadow=True, fontsize=20)
    plt.xticks()
    plt.tight_layout()
    plt.show()
def plot_single_model_violations(y_true_real, y_pred, n_max, n_min, model_name, title_suffix=""):

    color_true = '#2ecc71'
    color_pred = '#3498db'
    color_max = '#e74c3c'
    color_min = '#9b59b6'
    color_violation = '#e67e22'

    plt.figure(figsize=(24, 12), facecolor='#fafafa')
    ax = plt.subplot(111, facecolor='#ffffff')

    start_idx, end_idx = 330, 351

    steps = np.arange(len(y_true_real))
    y_true = y_true_real.flatten()
    y_p = y_pred.flatten()
    n_max_f = n_max.flatten()
    n_min_f = n_min.flatten()

    ax.fill_between(steps, n_min_f, n_max_f, color='#7f8c8d', alpha=0.08, label='Valid Zone')

    ax.plot(steps, n_max_f, ':', color=color_max, alpha=0.7, linewidth=3.0, label='Upper Bound (n_max)')
    ax.plot(steps, n_min_f, ':', color=color_min, alpha=0.7, linewidth=3.0, label='Lower Bound (n_min)')

    ax.plot(steps, y_true, label='True Values', color=color_true, alpha=0.6, linewidth=4.0)

    ax.plot(steps, y_p, label=f'Predictions ({model_name})', color=color_pred, linewidth=5.0, zorder=3)

    ax.set_title(f"Boundary Violations Analysis: {model_name} {title_suffix}", fontsize=32, fontweight='bold', pad=35,
                 color='#2c3e50')
    ax.set_xlabel("Time Steps", fontsize=24, labelpad=20, color='#34495e')
    ax.set_ylabel("Power, MW", fontsize=24, labelpad=20, color='#34495e')

    plt.xticks(fontsize=20, color='#7f8c8d')
    plt.yticks(fontsize=20, color='#7f8c8d')

    ax.grid(True, linestyle='--', alpha=0.3, color='#bdc3c7')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', framealpha=0.95,
              edgecolor='#e2e8f0', shadow=False, fontsize=18, labelspacing=0.6)

    plt.tight_layout()
    plt.show()
def plot_final_graph(output):

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 8), dpi=100)

    mae_before = np.mean(np.abs(output['Total_Error_Before']))
    max_before = output['Total_Error_Before'].max()
    min_before = output['Total_Error_Before'].min()

    mae_after = np.mean(np.abs(output['Total_Error_After']))
    max_after = output['Total_Error_After'].max()
    min_after = output['Total_Error_After'].min()

    stats_text = (
        f"  ★ PERFORMANCE METRICS ★\n\n"
        f"BEFORE OPTIMIZATION:\n"
        f"  • Mean Abs Error (MAE): {mae_before:.2f} MW\n"
        f"  • Max Overproduction:  {max_before:.2f} MW\n"
        f"  • Max Underproduction: {min_before:.2f} MW\n\n"
        f"AFTER OPTIMIZATION:\n"
        f"  • Mean Abs Error (MAE): {mae_after:.2f} MW\n"
        f"  • Max Overproduction:  {max_after:.2f} MW\n"
        f"  • Max Underproduction: {min_after:.2f} MW"
    )

    plt.plot(
        output.index,
        output['Total_Error_Before'],
        label='Imbalance BEFORE Optimization (Raw Forecast Error)',
        color='#e63946',
        linewidth=2.0,
        alpha=0.85
    )

    plt.plot(
        output.index,
        output['Total_Error_After'],
        label='Imbalance AFTER Optimization (Perfect Reconciled Balance)',
        color='#2a9d8f',
        linewidth=4.0,
        alpha=0.95
    )

    plt.axhline(0, color='#1d3557', linestyle='--', linewidth=1.5, alpha=0.8)

    plt.gca().text(
        0.98, 0.05,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=13,  # Крупный читаемый шрифт
        fontweight='medium',
        fontfamily='monospace',  # Моноширинный шрифт для выравнивания
        verticalalignment='bottom',
        horizontalalignment='right',  # ИСПРАВЛЕНО: выравнивание по правому краю
        bbox=dict(
            boxstyle='round,pad=0.8',
            facecolor='#f8f9fa',
            edgecolor='#cccccc',
            alpha=0.95
        )
    )

    # Настройка подписей, осей и легенды (крупно)
    plt.title('Optimization Effect: Eliminating Power Imbalance between Plant and Turbogenerators',
              fontsize=22, fontweight='bold', pad=25)
    plt.xlabel('Time Intervals (Row Index)', fontsize=16, labelpad=12)
    plt.ylabel('Imbalance Value (MW / Units)', fontsize=16, labelpad=12)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.ylim(-200, 50)

    plt.legend(fontsize=14, loc='upper right', frameon=True, shadow=True, facecolor='white')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()
