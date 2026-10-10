# Проверка 0.2.0

Python 3.13, Linux. 24 unittest-теста прошли. Проверены три календарных месяца, 8 demo-подстанций, идемпотентность, нелинейность холодовой нагрузки, отсутствие использования будущих выпусков погоды, метки перегрузки, временные блоки, модель и HTTP-ошибки.

Полный CLI-цикл генерация → обучение → сохранение модели выполнен. Реальный вызов /api/v1/predict с Open-Meteo: HTTP 200, 48 будущих часов, synthetic_demo. PostgreSQL и качество на реальных нагрузках не проверены. Ни эти тесты, ни метрики синтетики не доказывают точность предсказания реальных аварий.

```text
/data/smartgrid-venv/lib64/python3.13/site-packages/fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
  from starlette.testclient import TestClient as TestClient  # noqa
test_health_docs (test_stage1.ApiTests.test_health_docs) ... ok
test_validation_no_predict (test_stage1.ApiTests.test_validation_no_predict) ... ok
test_csv_and_upsert (test_stage1.ImportTests.test_csv_and_upsert) ... ok
test_duplicates_rejected (test_stage1.ImportTests.test_duplicates_rejected) ... ok
test_empty_rejected (test_stage1.ImportTests.test_empty_rejected) ... ok
test_seasonality (test_stage1.ImportTests.test_seasonality) ... ok
test_days_validation (test_stage1.WeatherTests.test_days_validation) ... ok
test_invalid_arrays (test_stage1.WeatherTests.test_invalid_arrays) ... ok
test_parse_store (test_stage1.WeatherTests.test_parse_store) ... ok
test_upstream_error (test_stage1.WeatherTests.test_upstream_error) ... ok
test_cold_growth_is_nonlinear (test_stage2.Stage2Tests.test_cold_growth_is_nonlinear) ... ok
test_empty_data_rejected (test_stage2.Stage2Tests.test_empty_data_rejected) ... ok
test_future_archived_weather_rejected (test_stage2.Stage2Tests.test_future_archived_weather_rejected) ... ok
test_generator_is_idempotent (test_stage2.Stage2Tests.test_generator_is_idempotent) ... ok
test_incompatible_artifact (test_stage2.Stage2Tests.test_incompatible_artifact) ... ok
test_label_consistency (test_stage2.Stage2Tests.test_label_consistency) ... ok
test_missing_model (test_stage2.Stage2Tests.test_missing_model) ... ok
test_missing_weather_horizon (test_stage2.Stage2Tests.test_missing_weather_horizon) ... ok
test_predict_48_hours (test_stage2.Stage2Tests.test_predict_48_hours) ... ok
test_predict_invalid_inputs (test_stage2.Stage2Tests.test_predict_invalid_inputs) ... ok
test_russian_district (test_stage2.Stage2Tests.test_russian_district) ... ok
test_temporal_partitions_and_artifact (test_stage2.Stage2Tests.test_temporal_partitions_and_artifact) ... ok
test_three_calendar_months_and_eight_stations (test_stage2.Stage2Tests.test_three_calendar_months_and_eight_stations) ... ok
test_weather_provider_failure (test_stage2.Stage2Tests.test_weather_provider_failure) ... ok

----------------------------------------------------------------------
Ran 24 tests in 3.731s

OK

```
