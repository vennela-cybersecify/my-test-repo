import os
import re
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# ---------------- CONFIG ----------------
DATA_FOLDER = "filtered_stocks_data"  # folder containing per-ticker CSVs
PREDICTION_FOLDER = "backtest_results"
OUTPUT_FOLDER = "stoploss_reports_hard_25"
MIN_GAIN_THRESHOLD = 100  # Success = ₹100+ gain
FIXED_STOPLOSS = 25


def calculate_zerodha_charges(buy_price, sell_price, qty=1, intraday=True):
    """
    Calculate Zerodha brokerage + statutory charges.
    qty = number of shares (default = 1)
    intraday=True => MIS trade, intraday charges
    intraday=False => CNC trade, delivery charges
    """

    turnover = (buy_price + sell_price) * qty

    # --- Brokerage ---
    if intraday:
        brokerage = min(0.0003 * turnover, 20)  # 0.03% or Rs.20 whichever lower
    else:
        brokerage = 0.0  # No brokerage on delivery

    # --- STT (Securities Transaction Tax) ---
    if intraday:
        stt = 0.00025 * (sell_price * qty)  # 0.025% on sell side
    else:
        stt = 0.001 * turnover  # 0.1% total (buy + sell)

    # --- Exchange transaction charges ---
    exch = 0.0000345 * turnover  # NSE equity

    # --- GST ---
    gst = 0.18 * (brokerage + exch)

    # --- SEBI charges ---
    sebi = 0.000001 * turnover

    # --- Stamp duty (on buy side only, ~0.015%) ---
    stamp = 0.00015 * (buy_price * qty)

    total_charges = brokerage + stt + exch + gst + sebi + stamp
    return round(total_charges, 2)


# ---------------- HELPERS ----------------
def extract_tickers_from_report(file_path):
    tickers = []
    with open(file_path, "r") as f:
        for line in f:
            match = re.match(r"^\d+\.\s+([a-zA-Z0-9\-]+)", line.strip())
            if match:
                tickers.append(match.group(1).lower())
    return tickers


# ---------------- MAIN ----------------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
summary_rows = []

for file in os.listdir(PREDICTION_FOLDER):
    if not file.endswith(".txt"):
        continue

    match = re.search(r"\d{4}-\d{2}-\d{2}", file)
    if not match:
        print(f"❌ Skipping file with no valid date: {file}")
        continue

    prediction_date_str = match.group(0)
    prediction_date = datetime.strptime(prediction_date_str, "%Y-%m-%d")
    t1_date = prediction_date - timedelta(days=1)

    file_path = os.path.join(PREDICTION_FOLDER, file)
    tickers = extract_tickers_from_report(file_path)

    report_lines = [f"📉 ₹25 Fixed Stoploss Prediction & Validation for {prediction_date_str}\n"]

    total = 0
    gain_before_sl = 0
    gain_after_sl = 0
    all_results = []
    trades = []  # <-- FIX: keep track of trades for brokerage calc

    negative_before_sl = []
    negative_after_sl = []

    for idx, ticker in enumerate(tickers, 1):
        try:
            csv_path = os.path.join(DATA_FOLDER, f"{ticker.lower()}.csv")
            if not os.path.exists(csv_path):
                report_lines.append(f"{idx}. {ticker.upper()} | ⚠️ CSV file not found\n")
                continue

            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            day_row = df[df["timestamp"].dt.date == prediction_date.date()]

            if day_row.empty:
                report_lines.append(f"{idx}. {ticker.upper()} | ⚠️ No daily data\n")
                continue

            open_price = day_row['open'].iloc[0]
            entry_price = open_price
            sl_price = entry_price - FIXED_STOPLOSS
            close_price = day_row['close'].iloc[0]
            day_low = day_row['low'].iloc[0]

            sl_hit = day_low < sl_price
            exit_price = sl_price if sl_hit else close_price
            sl_hit_text = "❌ SL HIT" if sl_hit else "✅ NO SL HIT"

            gain_before = close_price - entry_price
            gain_after = exit_price - entry_price
            rr_ratio = "-" if FIXED_STOPLOSS == 0 else f"{gain_after / FIXED_STOPLOSS:.2f}x"

            total += 1
            if gain_before >= MIN_GAIN_THRESHOLD:
                gain_before_sl += 1
            if gain_after >= MIN_GAIN_THRESHOLD:
                gain_after_sl += 1

            if gain_before < 0:
                negative_before_sl.append(ticker.upper())
            if gain_after < 0:
                negative_after_sl.append(ticker.upper())

            report_lines.append(
                f"{idx}. {ticker.upper()} | Entry: ₹{entry_price:.2f} | SL: ₹{sl_price:.2f} | SL Risk: ₹{FIXED_STOPLOSS} Fixed Stoploss, {sl_hit_text}\n"
                f"    📌 Open: ₹{entry_price:.2f} | Close: ₹{close_price:.2f} | Exit: ₹{exit_price:.2f}\n"
                f"    💰 Gain Before SL: ₹{gain_before:.2f} | Gain After SL: ₹{gain_after:.2f} | R:R: {rr_ratio}\n"
            )

            all_results.append({
                "ticker": ticker.upper(),
                "gain_before_sl": gain_before,
                "gain_after_sl": gain_after
            })

            # Save trade for brokerage calculation
            trades.append({
                "ticker": ticker.upper(),
                "open": entry_price,
                "close": close_price,
                "exit": exit_price,
                "gain_after_sl": gain_after
            })

        except Exception as e:
            report_lines.append(f"{idx}. {ticker.upper()} | ❌ Error: {e}")

    if total > 0:
        gain_100_before = sum(1 for r in all_results if r["gain_before_sl"] >= MIN_GAIN_THRESHOLD)
        gain_100_after = sum(1 for r in all_results if r["gain_after_sl"] >= MIN_GAIN_THRESHOLD)
        total_gain_before = sum(r["gain_before_sl"] for r in all_results)
        total_gain_after = sum(r["gain_after_sl"] for r in all_results)

        report_lines.append("\n📊 Success Rate Summary")
        report_lines.append(f"🔹 Total Stocks Evaluated: {total}")
        report_lines.append(f"✅ Gain Before SL (₹{MIN_GAIN_THRESHOLD}+): {gain_100_before} / {total} = {gain_100_before / total * 100:.1f}%")
        report_lines.append(f"✅ Gain After ₹{FIXED_STOPLOSS} SL (₹{MIN_GAIN_THRESHOLD}+): {gain_100_after} / {total} = {gain_100_after / total * 100:.1f}%")
        report_lines.append(f"\n📌 Only gains ≥ ₹{MIN_GAIN_THRESHOLD} are counted as successful.")
        report_lines.append("\n📈 Total Gain Summary")
        report_lines.append(f"💹 Sum of Gains Before SL (All stocks): ₹{round(total_gain_before, 2)}")
        report_lines.append(f"💰 Sum of Gains After SL (All stocks): ₹{round(total_gain_after, 2)}")

        report_lines.append("\n🔻 Loss Summary")
        report_lines.append(f"📉 Stocks with Loss Before SL: {len(negative_before_sl)}")
        if negative_before_sl:
            report_lines.append("   → " + ", ".join(negative_before_sl))
        report_lines.append(f"📉 Stocks with Loss After SL: {len(negative_after_sl)}")
        if negative_after_sl:
            report_lines.append("   → " + ", ".join(negative_after_sl))

        sl_hits_hard = sum(1 for r in all_results if r["gain_after_sl"] < r["gain_before_sl"])
        max_dd_before = min((r["gain_before_sl"] for r in all_results), default=0)
        max_dd_after = min((r["gain_after_sl"] for r in all_results), default=0)

        # Recalculate net profit using Zerodha charges
        net_profit_no_sl = 0
        net_profit_hard_sl = 0

        for trade in trades:
            buy_price = trade["open"]
            sell_price_no_sl = trade["close"]
            sell_price_sl = trade["exit"]
            qty = 1

            charges_no_sl = calculate_zerodha_charges(buy_price, sell_price_no_sl, qty, intraday=True)
            charges_sl = calculate_zerodha_charges(buy_price, sell_price_sl, qty, intraday=True)

            net_profit_no_sl += (sell_price_no_sl - buy_price) * qty - charges_no_sl
            net_profit_hard_sl += (sell_price_sl - buy_price) * qty - charges_sl

        summary_rows.append({
            "Type": "Fixed_25_SL",
            "Prediction Date": prediction_date_str,
            "Total Stocks": total,
            "Stocks passed 100 Before SL": gain_100_before,
            "Stocks passed After SL": gain_100_after,
            "Win Rate Before SL (%)": f"{(gain_100_before / total * 100):.1f}%",
            "Win Rate After SL (%)": f"{(gain_100_after / total * 100):.1f}%",
            "Total Gain Before SL": round(total_gain_before, 2),
            "Total Gain After SL": round(total_gain_after, 2),
            "SL_Hits_Hard": sl_hits_hard,
            "MaxDrawdown_before SL": round(max_dd_before, 2),
            "MaxDrawdown_HardSL": round(max_dd_after, 2),
            "NetProfit_AfterCharges_NoSL": round(net_profit_no_sl, 2),
            "NetProfit_AfterCharges_HardSL": round(net_profit_hard_sl, 2)
        })

    output_path = os.path.join(OUTPUT_FOLDER, f"{prediction_date_str}_25rs_stoploss_prediction.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✅ Saved: {output_path}")


# ---------------- GENERIC SUMMARY FUNCTION ----------------
def make_summary(df, period_col, output_name, extra_cols=None):
    # 🔥 drop rows where period_col is NaN (prevents ALL__Hnan row)
    df = df.dropna(subset=[period_col])

    # --- Base aggregated table ---
    grouped = df.groupby(period_col).agg({
        "Prediction Date": pd.Series.nunique,   # ✅ unique days, not rows
        "Total Stocks": "sum",
        "Stocks passed 100 Before SL": "sum",
        "Stocks passed After SL": "sum",
        "Total Gain Before SL": "sum",
        "Total Gain After SL": "sum",
        "NetProfit_AfterCharges_NoSL": "sum",
        "NetProfit_AfterCharges_HardSL": "sum"
    }).rename(columns={
        "Prediction Date": "Days Count",
        "Total Stocks": "Total Stocks Evaluated",
        "Stocks passed 100 Before SL": "Total Passed 100 (Before SL)",
        "Stocks passed After SL": "Total Passed 100 (After SL)",
        "Total Gain Before SL": "Total Profit (No SL)",
        "Total Gain After SL": "Total Profit (Hard SL)",
        "NetProfit_AfterCharges_NoSL": "Net Profit After Charges (No SL)",
        "NetProfit_AfterCharges_HardSL": "Net Profit After Charges (Hard SL)"
    })

    # --- Win Rates ---
    grouped["Win Rate Before SL (%)"] = (
        grouped["Total Passed 100 (Before SL)"] / grouped["Total Stocks Evaluated"] * 100
    ).round(2)
    grouped["Win Rate After SL (%)"] = (
        grouped["Total Passed 100 (After SL)"] / grouped["Total Stocks Evaluated"] * 100
    ).round(2)

    # --- Expectancy (avg PnL per prediction) ---
    grouped["Expectancy (No SL)"] = (
        grouped["Total Profit (No SL)"] / grouped["Total Stocks Evaluated"]
    ).round(2)
    grouped["Expectancy (Hard SL)"] = (
        grouped["Total Profit (Hard SL)"] / grouped["Total Stocks Evaluated"]
    ).round(2)

    # --- Distribution (Big Gainers / Small Gainers / Losses) ---
    grouped["Big Gainers (No SL)"] = grouped["Total Passed 100 (Before SL)"].astype(int)
    grouped["Big Gainers (Hard SL)"] = grouped["Total Passed 100 (After SL)"].astype(int)
    grouped["Losses (No SL)"] = (
        grouped["Total Stocks Evaluated"] - grouped["Total Passed 100 (Before SL)"]
    ).astype(int)
    grouped["Losses (Hard SL)"] = (
        grouped["Total Stocks Evaluated"] - grouped["Total Passed 100 (After SL)"]
    ).astype(int)
    grouped["Small Gainers (No SL)"] = (
        grouped["Total Stocks Evaluated"] - grouped["Big Gainers (No SL)"] - grouped["Losses (No SL)"]
    )
    grouped["Small Gainers (Hard SL)"] = (
        grouped["Total Stocks Evaluated"] - grouped["Big Gainers (Hard SL)"] - grouped["Losses (Hard SL)"]
    )

    # --- Per-period extra stats (avg/best/worst, Sharpe/Sortino, Recovery, Profit Factor, % Green Days) ---
    extra_rows = []
    for period, subdf in df.groupby(period_col):
        daily_gains_no_sl = subdf.groupby("Prediction Date")["Total Gain Before SL"].sum()
        daily_gains_hard_sl = subdf.groupby("Prediction Date")["Total Gain After SL"].sum()

        avg_daily_no_sl = daily_gains_no_sl.mean()
        avg_daily_hard_sl = daily_gains_hard_sl.mean()
        best_day = daily_gains_no_sl.max()
        worst_day = daily_gains_no_sl.min()

        std_no = daily_gains_no_sl.std(ddof=0)
        std_hard = daily_gains_hard_sl.std(ddof=0)
        sharpe_no_sl = (avg_daily_no_sl / std_no) if (pd.notnull(std_no) and std_no > 0) else 0
        sharpe_hard_sl = (avg_daily_hard_sl / std_hard) if (pd.notnull(std_hard) and std_hard > 0) else 0

        downside_std_no = daily_gains_no_sl[daily_gains_no_sl < 0].std(ddof=0)
        downside_std_hard = daily_gains_hard_sl[daily_gains_hard_sl < 0].std(ddof=0)
        sortino_no_sl = (avg_daily_no_sl / downside_std_no) if (pd.notnull(downside_std_no) and downside_std_no > 0) else 0
        sortino_hard_sl = (avg_daily_hard_sl / downside_std_hard) if (pd.notnull(downside_std_hard) and downside_std_hard > 0) else 0

        recov_no_sl = daily_gains_no_sl.sum() / abs(worst_day) if worst_day < 0 else float("inf")
        worst_day_hard = daily_gains_hard_sl.min()
        recov_hard_sl = daily_gains_hard_sl.sum() / abs(worst_day_hard) if worst_day_hard < 0 else float("inf")

        gross_profit_no_sl = daily_gains_no_sl[daily_gains_no_sl > 0].sum()
        gross_loss_no_sl = abs(daily_gains_no_sl[daily_gains_no_sl < 0].sum())
        profit_factor_no_sl = (gross_profit_no_sl / gross_loss_no_sl) if gross_loss_no_sl > 0 else float("inf")

        gross_profit_hard_sl = daily_gains_hard_sl[daily_gains_hard_sl > 0].sum()
        gross_loss_hard_sl = abs(daily_gains_hard_sl[daily_gains_hard_sl < 0].sum())
        profit_factor_hard_sl = (gross_profit_hard_sl / gross_loss_hard_sl) if gross_loss_hard_sl > 0 else float("inf")

        green_days_no_sl = (daily_gains_no_sl > 0).sum()
        pct_green_no_sl = (green_days_no_sl / len(daily_gains_no_sl) * 100) if len(daily_gains_no_sl) > 0 else 0
        green_days_hard_sl = (daily_gains_hard_sl > 0).sum()
        pct_green_hard_sl = (green_days_hard_sl / len(daily_gains_hard_sl) * 100) if len(daily_gains_hard_sl) > 0 else 0

        extra_rows.append({
            period_col: period,
            "Avg Daily Profit (No SL)": round(avg_daily_no_sl, 2),
            "Avg Daily Profit (Hard SL)": round(avg_daily_hard_sl, 2),
            "Best Day Profit": round(best_day, 2),
            "Worst Day Profit": round(worst_day, 2),
            "Sharpe-like (No SL)": round(sharpe_no_sl, 2),
            "Sharpe-like (Hard SL)": round(sharpe_hard_sl, 2),
            "Sortino (No SL)": round(sortino_no_sl, 2),
            "Sortino (Hard SL)": round(sortino_hard_sl, 2),
            "Recovery Factor (No SL)": round(recov_no_sl, 2),
            "Recovery Factor (Hard SL)": round(recov_hard_sl, 2),
            "Profit Factor (No SL)": round(profit_factor_no_sl, 2),
            "Profit Factor (Hard SL)": round(profit_factor_hard_sl, 2),
            "% Green Days (No SL)": round(pct_green_no_sl, 1),
            "% Green Days (Hard SL)": round(pct_green_hard_sl, 1),
        })

    extra_df = pd.DataFrame(extra_rows).set_index(period_col)
    grouped = grouped.join(extra_df)

    # --- CAGR (based on total PnL over trading days; assumes 252 trading days/yr & start_cap=100000) ---
    def calc_cagr(total_profit, days_count):
        if days_count <= 1:
            return 0.0
        start_cap = 100000.0
        end_cap = start_cap + float(total_profit)
        years = float(days_count) / 252.0
        return round(((end_cap / start_cap) ** (1.0 / years) - 1.0) * 100.0, 2)

    grouped["CAGR (No SL)"] = [
        calc_cagr(tp, dc) for tp, dc in zip(grouped["Total Profit (No SL)"], grouped["Days Count"])
    ]
    grouped["CAGR (Hard SL)"] = [
        calc_cagr(tp, dc) for tp, dc in zip(grouped["Total Profit (Hard SL)"], grouped["Days Count"])
    ]

    # --- Add Overall Row (same fields as your earlier version) ---
    overall_row = pd.DataFrame([{
        period_col: "Overall",
        "Days Count": df["Prediction Date"].nunique(),
        "Total Stocks Evaluated": df["Total Stocks"].sum(),
        "Total Passed 100 (Before SL)": df["Stocks passed 100 Before SL"].sum(),
        "Total Passed 100 (After SL)": df["Stocks passed After SL"].sum(),
        "Total Profit (No SL)": df["Total Gain Before SL"].sum(),
        "Total Profit (Hard SL)": df["Total Gain After SL"].sum(),
        "Net Profit After Charges (No SL)": df["NetProfit_AfterCharges_NoSL"].sum(),
        "Net Profit After Charges (Hard SL)": df["NetProfit_AfterCharges_HardSL"].sum(),
        "Win Rate Before SL (%)": round(df["Stocks passed 100 Before SL"].sum() / df["Total Stocks"].sum() * 100, 2),
        "Win Rate After SL (%)": round(df["Stocks passed After SL"].sum() / df["Total Stocks"].sum() * 100, 2),
    }]).set_index(period_col)

    grouped = pd.concat([grouped, overall_row])

    # --- Save ---
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    grouped.to_csv(output_path)
    print(f"📊 {period_col} summary saved to {output_path}")


# ---------------- DAILY SUMMARY ----------------
def make_daily_summary(df, output_name="daily_summary.csv"):
    daily = df.groupby("Prediction Date").agg({
        "Total Stocks": "sum",
        "Stocks passed 100 Before SL": "sum",
        "Stocks passed After SL": "sum",
        "Total Gain Before SL": "sum",
        "Total Gain After SL": "sum",
        "SL_Hits_Hard": "sum",
        "MaxDrawdown_before SL": "min",
        "MaxDrawdown_HardSL": "min",
        "NetProfit_AfterCharges_NoSL": "sum",
        "NetProfit_AfterCharges_HardSL": "sum"
    }).rename(columns={
        "Total Stocks": "Total Stocks Evaluated",
        "Stocks passed 100 Before SL": "Total Passed 100 (Before SL)",
        "Stocks passed After SL": "Total Passed 100 (After SL)",
        "Total Gain Before SL": "Total Profit (No SL)",
        "Total Gain After SL": "Total Profit (Hard SL)",
        "NetProfit_AfterCharges_NoSL": "Net Profit After Charges (No SL)",
        "NetProfit_AfterCharges_HardSL": "Net Profit After Charges (Hard SL)"
    })

    # --- Win Rates ---
    daily["Win Rate Before SL (%)"] = (
        daily["Total Passed 100 (Before SL)"] / daily["Total Stocks Evaluated"] * 100
    ).round(2)
    daily["Win Rate After SL (%)"] = (
        daily["Total Passed 100 (After SL)"] / daily["Total Stocks Evaluated"] * 100
    ).round(2)

    # --- Expectancy ---
    daily["Expectancy (No SL)"] = (
        daily["Total Profit (No SL)"] / daily["Total Stocks Evaluated"]
    ).round(2)
    daily["Expectancy (Hard SL)"] = (
        daily["Total Profit (Hard SL)"] / daily["Total Stocks Evaluated"]
    ).round(2)

    # --- Distribution ---
    daily["Big Gainers (No SL)"] = daily["Total Passed 100 (Before SL)"].astype(int)
    daily["Big Gainers (Hard SL)"] = daily["Total Passed 100 (After SL)"].astype(int)
    daily["Losses (No SL)"] = (
        daily["Total Stocks Evaluated"] - daily["Total Passed 100 (Before SL)"]
    ).astype(int)
    daily["Losses (Hard SL)"] = (
        daily["Total Stocks Evaluated"] - daily["Total Passed 100 (After SL)"]
    ).astype(int)
    daily["Small Gainers (No SL)"] = (
        daily["Total Stocks Evaluated"] - daily["Big Gainers (No SL)"] - daily["Losses (No SL)"]
    )
    daily["Small Gainers (Hard SL)"] = (
        daily["Total Stocks Evaluated"] - daily["Big Gainers (Hard SL)"] - daily["Losses (Hard SL)"]
    )

    # --- CAGR helper ---
    def calc_cagr(total_profit, days_count):
        if days_count <= 1:
            return 0.0
        start_cap = 100000
        end_cap = start_cap + total_profit
        years = days_count / 252  # trading days
        return round(((end_cap / start_cap) ** (1 / years) - 1) * 100, 2)

    # --- Add Overall Row ---
    overall_row = pd.DataFrame([{
        "Prediction Date": "Overall",
        "Total Stocks Evaluated": df["Total Stocks"].sum(),
        "Total Passed 100 (Before SL)": df["Stocks passed 100 Before SL"].sum(),
        "Total Passed 100 (After SL)": df["Stocks passed After SL"].sum(),
        "Total Profit (No SL)": df["Total Gain Before SL"].sum(),
        "Total Profit (Hard SL)": df["Total Gain After SL"].sum(),
        "SL_Hits_Hard": df["SL_Hits_Hard"].sum(),
        "MaxDrawdown_before SL": df["MaxDrawdown_before SL"].min(),
        "MaxDrawdown_HardSL": df["MaxDrawdown_HardSL"].min(),
        "Net Profit After Charges (No SL)": df["NetProfit_AfterCharges_NoSL"].sum(),
        "Net Profit After Charges (Hard SL)": df["NetProfit_AfterCharges_HardSL"].sum(),
        "Win Rate Before SL (%)": round(df["Stocks passed 100 Before SL"].sum() / df["Total Stocks"].sum() * 100, 2),
        "Win Rate After SL (%)": round(df["Stocks passed After SL"].sum() / df["Total Stocks"].sum() * 100, 2),
        "Expectancy (No SL)": round(df["Total Gain Before SL"].sum() / df["Total Stocks"].sum(), 2),
        "Expectancy (Hard SL)": round(df["Total Gain After SL"].sum() / df["Total Stocks"].sum(), 2),
        "Big Gainers (No SL)": int(df["Stocks passed 100 Before SL"].sum()),
        "Big Gainers (Hard SL)": int(df["Stocks passed After SL"].sum()),
        "Losses (No SL)": int(df["Total Stocks"].sum() - df["Stocks passed 100 Before SL"].sum()),
        "Losses (Hard SL)": int(df["Total Stocks"].sum() - df["Stocks passed After SL"].sum()),
        "Small Gainers (No SL)": 0,
        "Small Gainers (Hard SL)": 0,
        "CAGR (No SL)": calc_cagr(df["Total Gain Before SL"].sum(), df["Prediction Date"].nunique()),
        "CAGR (Hard SL)": calc_cagr(df["Total Gain After SL"].sum(), df["Prediction Date"].nunique())
    }]).set_index("Prediction Date")

    daily = pd.concat([daily, overall_row])

    # --- Save ---
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    daily.to_csv(output_path)
    print(f"📊 Daily summary saved to {output_path}")

# ---------------- RUN SUMMARIES ----------------
summary_df = pd.DataFrame(summary_rows)

summary_df["Month"] = summary_df["Prediction Date"].str[:7]
summary_df["Quarter"] = summary_df["Prediction Date"].str[:4] + "_Q" + (
    (pd.to_datetime(summary_df["Prediction Date"], errors="coerce").dt.month - 1) // 3 + 1
).astype(str)
summary_df["HalfYear"] = summary_df["Prediction Date"].str[:4] + "_H" + (
    (pd.to_datetime(summary_df["Prediction Date"], errors="coerce").dt.month - 1) // 6 + 1
).astype(str)
summary_df["Year"] = summary_df["Prediction Date"].str[:4]

# New Week & Bi-Week
summary_df["Week"] = pd.to_datetime(summary_df["Prediction Date"], errors="coerce").dt.to_period("W-SUN").astype(str)
dates = pd.to_datetime(summary_df["Prediction Date"], errors="coerce")
week_numbers = dates.dt.isocalendar().week
biweek_numbers = ((week_numbers - 1) // 2 + 1).astype(str)
summary_df["BiWeek"] = dates.dt.year.astype(str) + "_B" + biweek_numbers

# Now run summaries
make_daily_summary(summary_df, "fixed_25_daily_summary.csv")
make_summary(summary_df, "Week", "fixed_25_weekly_summary.csv")
make_summary(summary_df, "BiWeek", "fixed_25_biweekly_summary.csv")
make_summary(summary_df, "Month", "fixed_25_monthly_summary.csv")
make_summary(summary_df, "Quarter", "fixed_25_quarterly_summary.csv")
make_summary(summary_df, "HalfYear", "fixed_25_halfyearly_summary.csv")
make_summary(summary_df, "Year", "fixed_25_yearly_summary.csv")



