-- Medallion schema for optional SQL Server integration (step 2b).
-- Run against database configured in config.yaml / SQLSERVER_CONN_STR.

CREATE TABLE dbo.bronze_satellite_raw (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    obs_date DATE NOT NULL,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    ndvi FLOAT NULL,
    evi FLOAT NULL,
    temperature_2m FLOAT NULL,
    total_precipitation FLOAT NULL,
    dewpoint_temperature_2m FLOAT NULL,
    burned_area FLOAT NULL,
    source_system VARCHAR(50) NOT NULL,
    ingested_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
);

CREATE TABLE dbo.silver_features_clean (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    obs_date DATE NOT NULL,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    ndvi FLOAT NULL,
    evi FLOAT NULL,
    temperature_2m FLOAT NULL,
    total_precipitation FLOAT NULL,
    dewpoint_temperature_2m FLOAT NULL,
    ndvi_lag7 FLOAT NULL,
    temp_7d_mean FLOAT NULL,
    precip_7d_sum FLOAT NULL,
    ndvi_delta7 FLOAT NULL,
    day_of_year INT NULL,
    season_idx INT NULL,
    fire_within_7d BIT NULL,
    transformed_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
);

CREATE TABLE dbo.gold_model_results (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    run_date DATE NOT NULL,
    model_name VARCHAR(20) NOT NULL,
    precision_score FLOAT NOT NULL,
    recall_score FLOAT NOT NULL,
    f1_score FLOAT NOT NULL,
    roc_auc FLOAT NOT NULL,
    pr_auc FLOAT NOT NULL,
    train_minutes FLOAT NULL,
    created_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
);
