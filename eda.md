# Exploratory Data Analysis (EDA) Summary

## 1. Dataset overview

The merged dataset used for EDA contains:
- 456,548 rows
- 15 columns
- No missing values across all features

Columns include:
- target: `num_orders`
- temporal: `week`
- item/center identifiers: `meal_id`, `center_id`
- price features: `checkout_price`, `base_price`
- promotion features: `emailer_for_promotion`, `homepage_featured`
- categorical context: `category`, `cuisine`, `city_code`, `region_code`, `center_type`, `op_area`

The dataset is already merged from the raw train + meal + center tables, so the core modeling view is ready for feature engineering.

---

## 2. Target variable behavior

The target variable, `num_orders`, is strongly right-skewed.

Observed summary:
- Mean: 261.9
- Median: 136.0
- Standard deviation: 395.9
- Min: 13
- Max: 24,299

This means the distribution is dominated by many low-demand observations with a long upper tail. A small number of very high orders can strongly influence model learning.

### Modeling implication
- Use a log transform such as `log1p(num_orders)` for linear models or for feature analysis.
- Consider models that handle skewed count targets well, such as tree ensembles, gradient boosting, or count-based methods.
- Evaluate using metrics that are robust to skewness, not only MAE.

---

## 3. Demand distribution pattern

The histogram of orders below 2,000 shows the distribution is heavily concentrated near low values and then tapers into a long-right tail. This confirms that demand is not normally distributed.

### Modeling implication
- Standard scaling alone is not enough; target transformation and possibly outlier handling should be considered.
- If using regression models, target transformation is likely important.
- If using tree-based models, target skewness is less critical, but robust validation remains essential.

---

## 4. Weekly demand trend

The weekly average demand time series shows clear variation across weeks, including noticeable peaks at some weeks. This strongly suggests the presence of weekly seasonality and possibly event-driven spikes.

Examples from the observed averages:
- Some weeks are well above the baseline, such as around weeks 48 and 60.
- Several weeks show a noticeable drop, such as week 62.

### Modeling implication
- Include `week` as a time-based feature.
- Add lag features and rolling window features for short-term demand history.
- Consider cyclical encoding for time-of-year effects if the horizon is long enough.
- A baseline model should be benchmarked against a weekly seasonal baseline before trying more complex models.

---

## 5. Promotion effects

Both promotion indicators show a positive relationship with demand.

Observed correlations with `num_orders`:
- `emailer_for_promotion`: +0.277
- `homepage_featured`: +0.294

This indicates that when these marketing flags are active, orders tend to rise on average.

### Modeling implication
- Keep both promotion variables in the model.
- Try interaction terms such as `emailer_for_promotion × homepage_featured`.
- If the model is time-aware, consider whether promotion effects decay after a few weeks.

---

## 6. Price sensitivity

The price variables are negatively related to demand.

Observed correlations:
- `checkout_price` vs `num_orders`: -0.282
- `base_price` vs `num_orders`: -0.222

These are meaningful negative relationships, meaning higher prices are associated with lower order volumes.

### Important note
- `checkout_price` and `base_price` are extremely highly correlated with each other (`0.953`).

### Modeling implication
- Do not include both variables without care, because they are nearly redundant.
- Prefer one of them, or derive a discount / margin feature from the pair.
- Example feature engineering ideas:
  - `discount = base_price - checkout_price`
  - `discount_ratio = checkout_price / base_price`

---

## 7. Category-level demand differences

Average demand varies substantially by category.

Highest-average-order categories:
- `Rice Bowl`: 624.8
- `Sandwich`: 529.8
- `Salad`: 383.2
- `Beverages`: 316.5

Lower-demand categories include:
- `Pasta`: 59.1
- `Biryani`: 30.7
- `Desert`: 66.3

### Modeling implication
- `category` is a high-signal categorical feature.
- One-hot or target encoding can help capture category-specific demand behavior.
- Category-specific seasonal patterns may exist, so category-by-week interactions may improve forecast quality.

---

## 8. Cuisine-level demand differences

Cuisine also has a strong effect on demand.

Average demand by cuisine:
- `Italian`: 359.3
- `Thai`: 276.4
- `Indian`: 229.0
- `Continental`: 164.5

### Modeling implication
- `cuisine` should be retained as a categorical feature.
- It may capture a substantial share of demand variation beyond item identity alone.

---

## 9. Center-type effects

Demand differs by center type.

Average demand by center type:
- `TYPE_B`: 318.9
- `TYPE_A`: 262.4
- `TYPE_C`: 206.7

### Modeling implication
- `center_type` likely encodes a meaningful operational or customer-base difference.
- It should be included in the model as an informative categorical feature.

---

## 10. Region-level demand differences

Regional effects are visible and meaningful.

Average demand by region code:
- `71`: 334.2
- `56`: 316.5
- `85`: 286.8
- `23`: 250.2
- `93`: 238.8
- `77`: 217.0
- `34`: 206.1
- `35`: 131.3

### Modeling implication
- `region_code` contains useful spatial signal.
- It may capture local demand patterns, customer mix, or logistical conditions.

---

## 11. Operational area relationship

`op_area` has a small but positive correlation with orders (`+0.177`). This suggests larger operational area is associated with somewhat higher demand, but the effect is weaker than the promotion or price features.

### Modeling implication
- Keep `op_area`, but do not expect it to be the dominant driver.
- It may work well in combination with region or center-type features.

---

## 12. Feature correlation observations

From the numeric correlation matrix:
- `checkout_price` and `base_price` are almost duplicates.
- Promotions and demand move together.
- `op_area` contributes a mild positive relationship.
- `city_code`, `region_code`, and `center_id` have limited direct linear association with `num_orders` but may still be useful as categorical group signals.

### Modeling implication
- Use feature selection carefully.
- For tree models, sparse categorical and interaction effects may be more useful than heavy linear dependency assumptions.

---

## 13. Data quality observations

The merged dataset has:
- no nulls
- consistent numeric and categorical fields
- no obvious missing-value handling required before modeling

This is a good sign because it reduces preprocessing complexity.

---

## 14. Recommendations for the forecasting model

### Strongest modeling directions
1. Treat `num_orders` as a skewed continuous/count target.
2. Use `week` as a temporal feature and test seasonality/lags.
3. Keep promotions, price signals, category, cuisine, center type, and region in the feature set.
4. Carefully handle price redundancy by deriving discount features rather than using both price fields directly.
5. Use validation that respects time order rather than random splitting.

### Practical feature engineering ideas
- `log1p(num_orders)` for target transformation
- `discount = base_price - checkout_price`
- `discount_ratio = checkout_price / base_price`
- rolling mean demand by item/center/week
- lagged weekly demand by `meal_id` and `center_id`
- category-week or cuisine-week interaction features
- one-hot/target encoding for categorical variables

### Suggested model strategy
- Start with a strong baseline using simple weekly averages or moving averages.
- Then compare:
  - tree-based regression models
  - gradient boosting models
  - linear models after log transformation
- Pick the model that performs best on time-based validation while staying stable on the demand distribution tail.

---

## 15. Final takeaway

The main story from the EDA is that demand is driven by a combination of:
- time/seasonality
- promotions
- pricing
- product category and cuisine
- center and region context

The dataset is clean and rich enough for forecasting, but the target is highly skewed and the price variables are redundant. The most important modeling decisions will be around target transformation, time feature engineering, and controlling for category/region demand mix.
