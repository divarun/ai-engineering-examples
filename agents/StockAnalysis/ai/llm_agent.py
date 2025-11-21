from textwrap import dedent
from crewai import Task, Crew, Agent
import pandas as pd
import json

def build_interpretation_agent(llm):
    """
    Build a Market Interpreter agent with a clear, objective role.
    """
    return Agent(
        role="Market Interpreter",
        goal="Interpret structured market data objectively and generate professional reports.",
        backstory=(
            "A technical analyst who explains market conditions clearly and concisely, "
            "including limitations when data is incomplete."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True
    )


def interpret(agent, summary, ohlcv_df):
    """
    Run a structured CrewAI workflow:
    1. Produce a professional technical sentiment report
    2. Convert the report into an actionable trade plan
    """

    # -------------------------
    # Prepare OHLCV JSON
    # -------------------------
    df_flat = ohlcv_df.copy()
    if isinstance(df_flat.columns, pd.MultiIndex):
        df_flat.columns = ["_".join(map(str, col)) for col in df_flat.columns]

    ohlcv_json = df_flat.to_dict(orient="records")

    # -------------------------
    # 0️⃣ OHLCV Data Task
    # -------------------------
    data_task = Task(
        name="OHLCV Data",
        description="Provide historical OHLCV data for reference by subsequent tasks.",
        agent=agent,
        expected_output=json.dumps(ohlcv_json)  # must be string
    )

    # -------------------------
    # 1️⃣ Technical Sentiment Report
    # -------------------------
    report_task = Task(
        name="Technical Sentiment Report",
        description=dedent(f"""
            Generate a **professional market interpretation report** using ONLY:
            • Candlestick patterns
            • Provided indicator summary (SMA50, SMA200, MACD, RSI, ATR, Bollinger Bands)

            Requirements:
            1. Trend assessment (SMA50 vs SMA200)
            2. Momentum evaluation (MACD, RSI)
            3. Conflicts or confluence between indicators
            4. Market sentiment label (Bullish / Bearish / Neutral / Inconclusive)
            5. Professional structured formatting

            **If any indicator data is missing or NaN:**
            - Clearly indicate which indicators are unavailable.
            - Explain why analysis may be inconclusive.
            - Provide recommendations or cautionary notes.

            **Report Format:**
            - Section 1: Indicator Status (table)
            - Section 2: Market Sentiment
            - Section 3: Trade Plan Recommendations (if possible)
            - Section 4: Summary / Next Steps / Disclaimer

            Do NOT fabricate values. Return a professional, clear, actionable report.
        """),
        agent=agent,
        context=[data_task],
        expected_output="A structured professional technical sentiment report."
    )

    # -------------------------
    # 2️⃣ Trade Plan Generation
    # -------------------------
    strategy_task = Task(
        name="Trade Plan Generation",
        description=dedent(f"""
            Using ONLY the final sentiment report, produce a clean **trade plan**:
            - Entry, Stop-Loss, Take-Profit from OHLCV JSON only
            - SL = strongest support in last 10–20 days
            - TP = strongest resistance in last 10–20 days
            - Risk/Reward ≥ 1.0
            - Provide explicit justification for each level

            **IMPORTANT:** 
            - If required numeric data (ATR, Close, support/resistance) is missing or invalid, 
              DO NOT fabricate numbers. 
            - Instead, return a clear note explaining why the trade plan cannot be generated.

            **Final Output:**
            - Return a clean, formatted Markdown block.
            - Include a "Trade Plan Status" section (Available / Unavailable) with justification.
            - No system/user prefixes.
        """),
        agent=agent,
        context=[report_task],
        expected_output="A professional trade plan with numeric levels or a clear note if unavailable."
    )

    # -------------------------
    # Crew execution
    # -------------------------
    crew = Crew(
        agents=[agent],
        tasks=[data_task, report_task, strategy_task],
        verbose=True
    )

    result = crew.kickoff()
    return result
