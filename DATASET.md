# Dataset description (how it was obtained)

**Name:** Residential Electricity Consumption: Dataset combining Multi-round Longitudinal Surveys and Energy Provider Data (RECON-SL), as titled in the dataset's own README.
**Source:** public (open-access) release by the dataset authors on IEEE DataPort: https://ieee-dataport.org/open-access/residential-electricity-consumption-dataset-sri-lanka-recon-sl
**Collected by:** the dataset authors, from households in several Sri Lankan cities served by LECO: monthly meter records (smart and non-smart meters) plus three survey waves, about 4,000 households.

## Files used in this project
| File | Use |
|---|---|
| `consumption_data/non_smart_meter/monthly_consumption.csv` | monthly kWh per household (101,575 rows, 25 months: Oct 2022 to Oct 2024) |
| `survey_data/wave_1/w1_household_information_and_history.csv` | house type, floor area, SEC class, meter type, household size, expenditure |
| `survey_data/wave_1/w1_room_roster.csv` | rooms, bedrooms, bulbs, fans, air conditioners |
| `survey_data/wave_1/w1_appliances.csv` | appliance counts and weekly usage hours |
| `survey_data/wave_1/w1_demographics.csv` | ages, employment, hours at home |
The 15-minute and 6-hour smart-meter files were not used.

## Why the data is not included in the ZIP
The full dataset is large and is distributed by its authors; download it from the source above and point `01_build_dataset.py --data_dir` at the folder holding the CSV files listed.

## Ethics
Anonymised household IDs only. No names, addresses or contact details are present or were collected. The project reports aggregate patterns and the demo application stores nothing.

## Cleaning applied (see `01_build_dataset.py`)
Consumption stored as text with thousands separators was converted to numbers; 590 missing and 2,332 zero readings were removed (vacant/disconnected homes); the top 0.1% of readings were dropped as probable commercial use or errors (101,575 -> 98,554 rows). The dataset README mentions 20 months, but the file contains 25.
