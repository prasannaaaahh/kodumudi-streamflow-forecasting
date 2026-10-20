# Data

## `raw/kodumudi\\\\\\\_daily.csv`

Daily hydro-climatological records for the catchment draining to the **Kodumudi** gauge on the **Cauvery** river
(CWC gauge id `5008`, station code `cauv\\\\\\\_kodum`), **1 Jan 1980 – 31 Dec 2015** (13,149 days, no gaps, no missing values).

|Column|Unit|Description|
|-|-|-|
|`date`|–|calendar day (ISO format)|
|`rainfall\\\\\\\_mm`|mm/day|daily rainfall|
|`discharge\\\\\\\_m3s`|m³/s|daily mean river discharge at the gauge|
|`temperature\\\\\\\_c`|°C|daily air temperature|
|`soil\\\\\\\_moisture\\\\\\\_kgm2`|kg/m²|daily soil moisture|

## Provenance

* The dataset was supplied by the course instructor as part of a university course project (one sheet per study catchment). This repository uses only the sheet for the Kodumudi/Cauvery catchment.
* The original upstream sources of each variable are not documented in the supplied file, so they are not claimed here.

## Cleaning performed

The original Excel sheet had two metadata rows above the header, a column name with a double space (`Discharge  m3/s`), and separate `Year`/`Month`/`Day` columns. These were converted to a single `date` column, snake\_case names, and units were moved into the documentation above. **No values were altered, imputed or removed.**

## Aggregation rules (used throughout the project)

* rainfall → **sum** over the period (it accumulates)
* discharge, temperature, soil moisture → **mean** over the period (they are states)

