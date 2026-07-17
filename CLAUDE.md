# CLAUDE.md

Standing context for this project. Read this before starting any task.

## What we are building

A two part system for onion farmers in the Nashik cluster of Maharashtra:

1. A forecasting engine that predicts the next period wholesale modal price (rupees per quintal) for a given mandi.
2. An agent on top of it that recommends which nearby mandi to sell in, and on which day, to maximise net profit after transport and storage costs.

The end output for a user is one sentence, for example: "Sell on Tuesday next week at Pimpalgaon mandi at about 2,110 rupees per quintal, roughly 230 more per quintal than selling locally today."

This is a student group project for an AI and Analytics course. The model benchmarking is the graded core of the work, so honest accuracy comparison matters more than a clever model.

## The data

* Onion price excel of daily onion prices. Update this line with the real filename once it is in the repo.
* Granularity: State, District, Market Name (mandi), Price Date, with Min, Max and Modal price.
* Range: 2023 to 2025 only. This is short history, plan around it.
* Target markets: example the Nashik cluster, primarily Lasalgaon, Pimpalgaon and Yeola. Confirm which markets actually have enough continuous data before committing.
* Target variable: Modal price for a future day.

Do not describe the columns back to me from assumption. Inspect the file first and report what is actually there.

## Known data issues to handle

* Missing dates where the mandi was closed. Reindex to a daily calendar and decide on a fill method.
* Duplicate rows and zero or absurd price values.
* Outlier spikes are normal for onion and are real, so handle them carefully rather than deleting all of them.
* The data has no latitude or longitude. Market towns will need geocoding later for the distance feature.

## Modelling approach

* Baseline first: seasonal naive (tomorrow = today, and same day last week). Every model must beat this.
* Classical: ARIMA or SARIMA, and Prophet.
* Machine learning: LightGBM as the likely workhorse, XGBoost as a second. Include the market as a feature so one global model covers all mandis.
* Deep learning: LSTM is optional and a stretch goal. With only two years of data it may not beat LightGBM, and that result is fine to report.

## Features

* Lags: 1, 3, 7, 14 days.
* Rolling mean and standard deviation: 7 and 14 day windows.
* Calendar: day of week, month, week of year, and a festival flag.
* Volatility: Max minus Min spread.
* Market identifier as a feature.
* Drop early rows where lags are undefined.

## Evaluation (the graded part)

* Split by time. Train on the earlier period, test on the later one. Never shuffle.
* Use walk forward validation where the train and test window rolls forward.
* Metrics: MAE, RMSE and MAPE.
* Produce one comparison table of every model on the same held out weeks.
* Add a small ablation showing whether the festival flag and the spread feature actually help.
* Save the winning model to disk. That becomes the served model for the agent.

## The agent layer

* Wrap the saved model as a simple function, forecast(market, date).
* Geocode the market towns to latitude and longitude, then filter to mandis within 50 km of the farmer using the haversine formula.
* Cost functions: transport = distance times a per km rate, storage = days held times a per day rate. Start these as clear assumptions.
* Decision rule: net realizable price = forecast price minus transport cost minus storage cost. Compute for every mandi and day, then take the argmax.
* Add a minimum margin guardrail so it does not recommend a long haul for a tiny gain.
* Simple Streamlit chat as the interface. The natural language reply can come from an LLM API.

## Sensitivity check

Move the transport and storage rates, for example fuel up by a set amount, and check whether the recommended mandi flips. This is part of the final deliverable.

## How I want you to work

* Python.
* Work in stages and stop for review after each one. Do the cleaning, show the result, then features, then models. Do not build the whole pipeline in a single pass.
* Keep explanations brief.
* No em dashes in any text you write. Use commas, hyphens or short sentences instead.
* Keep the repo tidy: a clear structure for data, notebooks or scripts, saved models, and the app.

