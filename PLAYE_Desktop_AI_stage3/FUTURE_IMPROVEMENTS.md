# PLAYE Pro Lab — Future Improvements Roadmap

Этот документ аккумулирует приоритетные улучшения, чтобы развить лабораторию
до уровня production/pro (в т.ч. для криминалистических сценариев).

## Уже реализовано (baseline)

- Request tracing через `X-Request-ID` в desktop и cloud backend.
- Базовый forensic audit log в формате JSONL (`models-data/audit/events.jsonl`).
- Фиксация `input_sha256`/`output_sha256`, длительности (`duration_ms`) и статуса операции.
- Оптимизация hot-path: SHA-256 и расширенный audit payload считаются только при включённом audit-режиме.
- Кэширование служебных метаданных (manifest meta cache и audit path cache) для снижения накладных расходов на запрос.

## 1) Архитектура и пайплайн

### 1.1 Unified backend API
- Объединить desktop `backend/server.py` и cloud-роуты `backend/app/api/routes.py` в единый контракт API.
- Согласовать форматы ответов (`request_id`, `status`, `error`, `result`).
- Единая модель job execution: sync + async режимы с `task_id`.

### 1.2 Stage-based processing graph
- Перейти от отдельных endpoint-операций к графу стадий:
  1. ingest
  2. detect/track
  3. enhance/restore
  4. QA metrics
  5. export/report
- Для видео: поддержка temporal consistency и track-aware обработки.

## 2) Forensic-grade доказуемость

### 2.1 Provenance / audit trail
- На каждую операцию сохранять:
  - `request_id`
  - `operator`
  - `input_sha256`
  - `output_sha256`
  - `model_name` + `model_version`
  - параметры запуска
  - timestamp UTC

### 2.2 Chain-of-custody reports
- Экспортировать отчёт (JSON + PDF):
  - история трансформаций,
  - контрольные суммы,
  - версии моделей,
  - предупреждение о допустимости/ограничениях результатов.

## 3) Модели и качество

### 3.1 Model portfolio
- Face restoration: RestoreFormer / CodeFormer / GFPGAN (режимы quality vs identity).
- Super-resolution: Real-ESRGAN x2/x4 + варианты для видео.
- Denoise/Deblur: NAFNet + специализированные deblur-модели.
- Detection: YOLOv8-face/RetinaFace + object detection для контекста сцены.

### 3.2 Quality gates
- Benchmark наборы (golden set) для фото и видео.
- Метрики: PSNR, SSIM, LPIPS, NIQE/BRISQUE, temporal flicker.
- Запрет публикации релиза при деградации метрик выше заданного порога.

## 4) Производительность

### 4.1 GPU efficiency
- tiled inference,
- mixed precision (fp16/bf16),
- TensorRT/ONNX Runtime для hot paths,
- preloading моделей и warmup.

### 4.2 Queue orchestration
- Redis + Celery/RQ для асинхронного батч-процессинга,
- retries/backoff,
- SLA по времени обработки и мониторинг очередей.

## 5) UX и операторские режимы

- Presets:
  - `Forensic Safe` (минимальная агрессивность изменений),
  - `Balanced`,
  - `Presentation`.
- Базовый preset selector уже добавлен в web UI и прокидывается в backend queue params; далее — связать пресеты с полноценным stage-graph.
- Сравнение before/after (slider, zoom sync, diff heatmap).
- Прозрачный лог действий оператора в UI.

## 6) Безопасность и соответствие

- Ролевой доступ (RBAC),
- защита API ключей и secrets,
- шифрование артефактов at-rest,
- настраиваемая политика хранения и удаления материалов.

## Прогресс по глобальному плану (оценка)

- **Phase 1:** ~100% (request tracing + audit trail уже есть; внедрён unified response schema, базовые `/job/submit` + `/job/{task_id}/status`, frontend polling integration, расширенная обработка статусов job polling + unified state mapping (`queued/running/retry/done/failed/canceled`) + `is_final/poll_after_ms` hints for stable frontend polling, operator-driven cancel endpoint + UI control, pause/resume queue controls in UI + manual retry/clear terminal queue controls, baseline `ci:smoke` checks и workflow `ci-smoke` в GitHub Actions, configurable pipeline params, backend param validation тестами и scene-aware detect_objects params, плюс worker-based temporal denoise в frontend и preset-aware queue params, осталось полноценный queued orchestration под нагрузкой и расширенные quality checks в CI).
- **Phase 2:** ~89% (усилен video temporal pipeline: ONNX worker + server-side video jobs, восстановлена backward compatibility API-эндпоинтов и улучшена обработка batch/video параметров; остались оптимизации качества/производительности и полный UX-цикл export/report).
- **Phase 3:** ~64% (добавлены multi-GPU primitives, batch API и enterprise foundation; дополнительно усилена аутентификация через проверку сессии/exp и контроль регистрации пользователей, расширен reporting suite (`/api/enterprise/reports/*` JSON/CSV + manifest), добавлены timeseries-отчёты активности (JSON/CSV) для аналитики нагрузки, а также введён глобальный RBAC middleware с path-template matching; остаются production hardening, observability, governance и полноценные e2e тесты).

## Реализация по этапам

### Phase 1 (сделать в первую очередь)
1. Unified API schema + request tracing.
2. Базовый forensic audit log.
3. Job orchestration (queued/sync).
4. Базовые quality checks в CI.

### Phase 2
1. Video temporal pipeline.
2. Расширенный модельный каталог и пресеты.
3. Автоматические forensic reports.

### Phase 3
1. Полноценная model governance система.
2. Многомашинная/многогпу оркестрация.
3. Продвинутые explainability/traceability инструменты.
