// Builds the assignment submission document.
// Run:  node docs/build_doc.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, PageBreak,
  LevelFormat, convertInchesToTwip,
} = require("docx");

const FIG = path.join(__dirname, "figures");
const BRAND = "14493B", GREEN = "12924E", AMBER = "D18F00";
const INK = "1A1A1A", MUTED = "6B6A65", ROW = "F4F6F3";
const USABLE = 9746; // DXA text width with 0.75" margins on A4

const pic = (file, w, h) => new ImageRun({
  type: "png", data: fs.readFileSync(path.join(FIG, file)),
  transformation: { width: w, height: h },
});

const img = (file, w, h) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 },
  children: [pic(file, w, h)],
});

const imgPair = (a, b) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 },
  children: [pic(a[0], a[1], a[2]), new TextRun({ text: "  " }), pic(b[0], b[1], b[2])],
});

const caption = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 130 },
  children: [new TextRun({ text: t, italics: true, size: 15, color: MUTED })],
});

const h2 = (t) => new Paragraph({
  spacing: { before: 170, after: 60 },
  children: [new TextRun({ text: t, bold: true, size: 22, color: BRAND })],
});

const p = (t, opts = {}) => new Paragraph({
  spacing: { after: 80 }, alignment: AlignmentType.JUSTIFIED,
  children: [new TextRun({ text: t, size: 19, color: INK, ...opts })],
});

const bullet = (t) => new Paragraph({
  numbering: { reference: "dots", level: 0 }, spacing: { after: 60 },
  children: [new TextRun({ text: t, size: 19, color: INK })],
});

const cell = (text, { bold = false, head = false, shade = null, align = AlignmentType.LEFT, width }) =>
  new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, fill: BRAND, color: "auto" }
      : shade ? { type: ShadingType.CLEAR, fill: shade, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({
        text, bold: bold || head, size: 18,
        color: head ? "FFFFFF" : INK,
      })],
    })],
  });

function accuracyTable(rows) {
  const W = [3446, 1450, 1450, 1700, 1700];
  const header = new TableRow({
    tableHeader: true,
    children: ["Method", "1-day MAE", "1-day MAPE %", "7-day MAE", "7-day MAPE %"].map((t, i) =>
      cell(t, { head: true, width: W[i], align: i ? AlignmentType.RIGHT : AlignmentType.LEFT })),
  });
  const body = rows.map((r, ri) => new TableRow({
    children: r.map((t, i) => cell(String(t), {
      width: W[i], bold: ri === 0, shade: ri === 0 ? "E6F0E9" : (ri % 2 ? ROW : null),
      align: i ? AlignmentType.RIGHT : AlignmentType.LEFT,
    })),
  }));
  return new Table({
    columnWidths: W, width: { size: USABLE, type: WidthType.DXA },
    rows: [header, ...body],
    borders: ["top", "bottom", "left", "right", "insideHorizontal", "insideVertical"]
      .reduce((a, k) => (a[k] = { style: BorderStyle.SINGLE, size: 2, color: "D8DCD6" }, a), {}),
  });
}

// method, 1-day MAE, 1-day MAPE, 7-day MAE, 7-day MAPE  (ordered by 1-day MAE)
const ACC = [
  ["Naive (last price)", 165.2, 6.50, 388.7, 16.01],
  ["ARIMA(1,1,1)", 167.2, 6.63, 388.0, 16.03],
  ["SARIMA(1,1,1)(1,0,0,7)", 168.3, 6.66, 389.0, 16.07],
  ["LightGBM (return target)", 168.7, 6.70, 448.0, 18.00],
  ["7-day moving average", 239.3, 9.85, 447.1, 18.94],
  ["LightGBM (level target)", 282.4, 10.86, 520.9, 22.99],
  ["XGBoost (level target)", 334.7, 12.10, 605.6, 24.63],
  ["Prophet", 610.3, 27.77, 686.4, 31.73],
];

const doc = new Document({
  numbering: {
    config: [{
      reference: "dots",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 340, hanging: 200 } } },
      }],
    }],
  },
  sections: [{
    properties: { page: { margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 } } },
    children: [
      new Paragraph({
        spacing: { after: 40 },
        children: [new TextRun({ text: "AI and Analytics for Managers  |  Group Assignment", size: 18, color: MUTED })],
      }),
      new Paragraph({
        spacing: { after: 40 },
        children: [new TextRun({ text: "Group [NAME]:  [Member names and roll numbers]", size: 20, bold: true, color: INK })],
      }),
      new Paragraph({
        spacing: { after: 200 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: AMBER } },
        children: [new TextRun({
          text: "Mandi Onion Sell Advisor: deciding where and when a Nashik farmer should sell",
          size: 28, bold: true, color: BRAND,
        })],
      }),

      h2("1.  The process and its value"),
      p("When an onion farmer in the Nashik cluster is ready to sell, two decisions follow: which mandi (APMC market yard) to cart the crop to, and whether to sell today or hold. Nashik handles a large share of India's onion trade, and on any given day the modal price differs sharply between yards only 20 to 40 km apart. In our data, on a single day Lasalgaon quoted Rs 1,600 per quintal while Yeola quoted Rs 1,200, a gap of 33 per cent."),
      p("Both the gap and the costs are real. Carting 40 km is not free, and holding stock adds storage loss and delays cash. A farmer with 100 quintal who picks the wrong yard can give up roughly Rs 30,000 on one trip. Today that choice is usually made on word of mouth, a trader's quote, or yesterday's rate heard in the village, none of which nets transport and storage against the price on offer. Automating it turns scattered public price data into one costed answer: sell at this mandi, for roughly this net price, or wait."),

      h2("2.  Current state and positioning"),
      p("Mandi prices are published daily by Agmarknet and eNAM, and several apps re-display those tables. A farmer can look up a price, but the table stops at the raw quote. It does not account for how far the farmer is from each yard, what freight costs over that distance, what storage costs if they wait, or what the price may be when they arrive. The farmer does that arithmetic mentally, if at all, and usually for only the one or two yards they know."),
      p("Our prototype extends these solutions rather than replicating them. The price-lookup step is already solved and we assume it. We add the decision layer above it: a benchmarked forecasting engine, a cost model that nets transport and storage against each yard from the farmer's own GPS location, and an optimisation ranking every mandi within a 72 km catchment by net realisable price. It also adds a confidence guardrail (section 5) that declines to recommend waiting unless the forecast supports it. The contribution is the costed decision and the guardrail, not the price data."),
      img("fig1_architecture.png", 520, 197),
      caption("Figure 1.  End to end flow, from open price data to a single costed recommendation."),


      h2("3.  Approach"),
      p("We split the problem in two. The first half is a forecasting engine that predicts the modal price for a mandi; every model had to be measured against a seasonal-naive baseline (\"tomorrow's price equals today's\") on identical held-out weeks, using expanding-window walk-forward validation rather than a shuffled split, at two horizons: one day, and seven days, the horizon the sell-or-wait decision needs."),
      p("The second half is a decision agent, which uses no machine learning. It is a deterministic rule engine: haversine distance from the farmer's GPS to each geocoded mandi, a road-circuity adjustment, explicit cost functions, and an argmax over net realisable price with two guardrails. A language model is used only to phrase the final answer, never to forecast or decide."),

      h2("4.  Input data"),
      p("A Kaggle daily mandi price dataset in Agmarknet format: 298,658 rows of all-India onion prices from 6 June 2023 to 11 June 2025, with State, District, Market, Variety, Grade, Min, Max and Modal price and Date."),
      p("Cleaning was substantial. Variety and Grade meant one market could have several rows on the same day, so we collapsed these to one modal price per market per day (298,653 to 279,404 rows). Zeros in Min and Max were missing-value sentinels, not real prices (253 rows), producing apparent Min above Max. Five rows were data-entry errors around 100 times the real value. Market names are not unique across states, so markets were keyed as State plus District plus Market. All of January 2025 is absent, and June and December 2024 are heavily under-collected. We scoped to the six Nashik mandis with the most continuous history (Nasik, Pimpalgaon, Lasalgaon, Manmad, Chandvad, Yeola; 2,592 market-days) and geocoded each APMC yard manually, as the dataset carries no coordinates."),

      h2("5.  Output and accuracy"),
      p("The prototype outputs a ranked table of every mandi in the catchment with distance, today's price, transport cost, net price now, the seven-day forecast and net price if held, plus one recommendation in English and Marathi, and a map. Accuracy pooled over walk-forward folds on identical held-out rows:"),
      accuracyTable(ACC.map(r => [r[0], r[1].toFixed(1), r[2].toFixed(2), r[3].toFixed(1), r[4].toFixed(2)])),
      caption("Table 1.  Accuracy at both horizons, pooled over walk-forward folds (1,050 rows / 10 folds at one day; 1,009 rows / 9 folds at seven days). Prices in rupees per quintal. Lower is better. Naive is best at one day; ARIMA only ties it at seven days."),
      p("No model beats the naive baseline. ARIMA ties it and was chosen as the served model because it also returns a confidence band. Because the seven-day forecast is close to flat with a wide band, the agent's timing advice is gated: it recommends holding only when the pessimistic end of the forecast, net of costs, still beats selling today by the minimum margin, so in practice it almost always recommends selling now. A feature ablation showed the festival-day flag did not improve accuracy and slightly worsened it, while the high-low spread feature gave only a marginal gain."),

      h2("6.  Models tested"),
      p("Seasonal naive (today's price, and same day last week); 7-day moving average; ARIMA(1,1,1); SARIMA(1,1,1)(1,0,0,7); Prophet; LightGBM on a price-level target; XGBoost on a price-level target; and LightGBM on a log-return target with mean-reversion and momentum features. All eight were scored on the identical folds in Tables 1 and 2, plus a feature ablation on the festival flag and the price spread."),

      h2("7.  Prototype"),
      p("A Streamlit web app. The farmer pastes a Google Maps link or types a latitude and longitude, enters a quantity, and receives a ranked set of mandis, a map, and a recommendation in English and Marathi. Cost assumptions are exposed as sliders, so the app doubles as a live sensitivity tool. A follow-up chat answers questions using only the computed numbers."),
      imgPair(["shot_reco.png", 300, 198], ["shot_chat.png", 300, 200]),
      caption("Figure 2.  Left: recommendation for a farmer 56 km from Lasalgaon, in English and Marathi. Right: follow-up chat asked in Hinglish, answered from the computed figures only (Nasik Rs 1,087 net against Lasalgaon Rs 1,382)."),
      img("shot_map.png", 330, 219),
      caption("Figure 3.  Mandis in the catchment, with the farmer's own location marked."),

      h2("8.  Tools used"),
      p("Python 3.11 (Miniconda, conda-forge). statsmodels for ARIMA and SARIMA; Prophet; LightGBM; XGBoost; scikit-learn for metrics; pandas for data preparation; joblib to persist the served model; matplotlib for figures. Streamlit for the prototype interface. Claude Opus 4.8 through the Anthropic API for the bilingual reply layer, and Claude Code as the development environment."),

      h2("9.  Benchmark comparison"),
      p("Our primary benchmark is the seasonal-naive rule, the standard reference baseline in forecasting. Our models match it but do not beat it."),
      p("We also compared against an independent implementation of the same problem, built by another team on the same dataset, which reported a LightGBM change-target model beating naive at seven days (195 against 203 MAE). We reproduced that result exactly and confirmed it was free of leakage. Their evaluation, however, used a single train and test split. Re-run under walk-forward validation across ten months, the same model won only 5 of 9 monthly folds and lost on the pooled average, dragged down by a large error in the volatile October 2024 festival period. The conclusion depends on the validation design, not the model."),
      p("Published work on short-horizon Indian mandi price forecasting commonly reports MAPE in roughly the 5 to 15 per cent band, which brackets our 6.5 per cent at one day and 16 per cent at seven days. That comparison is indicative only, as such studies typically use arrivals and weather alongside price, on different markets and periods. What we replicate is the forecasting and costed sell-decision task on Agmarknet-format data for six mandis; what we do not replicate is work using arrivals, weather and stock data, the actual drivers of onion prices, which this dataset lacks."),
      img("fig2_weekly_forecast.png", 500, 200),
      caption("Figure 4.  Weekly modal price at Lasalgaon against seven-day-ahead forecasts. The dashed naive line sits almost exactly on ARIMA, while LightGBM overshoots the October 2024 spike."),


      h2("10.  Key challenges"),
      bullet("Data quality was the bulk of the work: several variety and grade rows per market-day, zeros used as missing-value sentinels, entries around 100 times the real value, market names that are not unique across states, and an entire month absent from the dataset."),
      bullet("Our machine-learning models did not beat a naive rule. The disciplined response was to report that rather than to search for a test window that flattered them."),
      bullet("Designing around an unreliable forecast. Because seven-day forecasts carry a wide uncertainty band, we built a confidence gate so the agent does not advise a farmer to wait on a signal that is not there."),
      bullet("Environment failures cost real time: a numpy and MKL incompatibility silently crashed every linear-algebra call, and Prophet's Stan backend needed manual repair, before any model could be fitted."),

      h2("11.  Key learnings"),
      bullet("A strong baseline is the real test of a model. Daily mandi prices behave close to a random walk, so \"tomorrow equals today\" is hard to beat, and any model claiming to beat it must be checked against it."),
      bullet("Validation design can change the conclusion. The same model on the same data looked like a winner on a single holdout and a loser under walk-forward."),
      bullet("Missing drivers cap accuracy more than model choice does. Without arrivals, weather or stock data, no amount of tuning closed the gap; the ceiling is set by the data."),
      bullet("Scope the product to what the model supports. Ranking where to sell is reliable and needs no good forecast; advising when to sell is not, so we gated it rather than shipping false confidence."),
    ],
  }],
});

Packer.toBuffer(doc).then((b) => {
  const out = path.join(__dirname, "Group_Submission_Mandi_Onion_Sell_Advisor.docx");
  fs.writeFileSync(out, b);
  console.log("wrote " + out);
});
