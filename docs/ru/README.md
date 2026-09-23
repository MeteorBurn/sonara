# SONARA — документация на русском языке

SONARA — библиотека анализа музыки на Rust с Python-интерфейсом. Она извлекает
численные характеристики аудио: темп, атаки, спектральный состав, тембр,
гармонию, громкость и структуру. Поверх этих измерений работают перцептивные
эвристики и отдельные обученные модели. Это разные уровни достоверности:
например, спектральный центроид измеряет распределение частот, а `valence`
лишь приближённо описывает эмоциональную окраску.

Справочник подготовлен по исходникам **SONARA 0.3.7**, ревизия `0c2a4b3`,
23 сентября 2026 года. Он охватывает собственные модули Rust-ядра, Python-привязок
и Python-пакета. Внешние зависимости описаны по их роли, без попытки заменить
их документацию. Это справочник реализации и основных семейств API,
а не обещание полной совместимости с librosa или переведённый rustdoc каждого
вспомогательного символа.

## Как читать

| Раздел | Что в нём искать |
|---|---|
| [API и конвейер](api.md) | Установка, примеры Rust/Python, режимы и полный реестр групп, формы данных, ошибки, JSON, кеш и augment |
| [Численное ядро и DSP](dsp.md) | Загрузка/ресэмплинг, FFT/STFT/CQT, фильтры, MFCC/chroma, pitch, разделение, эффекты и последовательности |
| [Ритм, темп и гармония](rhythm-tonality.md) | Onset, ACF/DP, beat grid, rhythmic regularity, tempogram, размер, HPCP, аккорды, тональность, диссонанс |
| [Признаки, структура и модели](descriptors-models.md) | LUFS/true peak, энергия и настроение, структура, similarity/fingerprint, вокальность, жанр и aggression |

Если задача — проанализировать музыкальную библиотеку, начните с API.
Если нужно понять, почему изменился BPM или что именно означает оценка,
перейдите к соответствующему алгоритму. В каждой главе есть ссылки на код,
параметры и ограничения: название известного метода не означает, что здесь
реализованы все варианты этого метода из научной литературы.

## Архитектура и путь сигнала

```text
Файл ── декодирование, моно, ресэмплинг ─┐
                                      ├─ analyze_signal + AnalysisConfig
Массив f32 + фактический sample rate ──┘
    │
    ├─ кадровый проход: Hann → FFT → модуль / мощность
    │     ├─ mel → log-mel → MFCC и атаки → темп / удары
    │     ├─ chroma → тональность
    │     ├─ спектральные пики → HPCP / аккорды / диссонанс
    │     └─ центроид / bandwidth / rolloff / flatness / contrast
    ├─ временные измерения: RMS, ZCR, K-weighting / LUFS
    ├─ опциональные траектории, структура, fingerprint, модели
    └─ TrackAnalysis + provenance
          ├─ типизированный результат Rust
          └─ PyO3 → словарь Python → TrackAnalysis(dict)
```

Объединённый анализ переиспользует FFT, а не вызывает независимо все
standalone-функции. Отдельные DSP-модули нужны для собственных цепочек обработки.
Граф вычислительных зависимостей шире выдаваемого результата: модель может
потребовать embedding, не публикуя его наружу. Подробнее — [конвейер](api.md).

Rust-ядро не зависит от PyO3. Крейтом `sonara-python` реализовано преобразование
массивов, конфигурации, ошибок и результатов, но не второе DSP-ядро. Python-фасад
добавляет удобный словарь результата, сравнение, загрузку/обучение небольших
моделей и отображение.

## Карта всех модулей Rust-ядра

Пути ниже относятся к собственным исходникам библиотеки. Файлы `mod.rs`
объединяют подмодули; это не отдельные алгоритмы.

| Модуль / исходник | Назначение | Описание |
|---|---|---|
| [lib.rs](../../sonara/src/lib.rs) | Объявление модулей, feature gates и корневые экспорты типов/ошибок | [API](api.md) |
| [types.rs](../../sonara/src/types.rs) | `Float = f32`, массивы аудио и спектра, параметры окон/нормализации | [DSP](dsp.md) |
| [error.rs](../../sonara/src/error.rs) | `SonaraError`, тип `Result`, разделение входных, файловых и численных ошибок | [API](api.md) |
| [analyze.rs](../../sonara/src/analyze.rs) | Реестр признаков, режимы, объединённый проход, provenance, batch, augment | [API](api.md) |
| [core/mod.rs](../../sonara/src/core/mod.rs) | Организация численного ядра | [DSP](dsp.md) |
| [core/audio.rs](../../sonara/src/core/audio.rs) | Декодирование, метаданные, ресэмплинг, генераторы, автокорреляция, LPC, μ-law | [DSP](dsp.md) |
| [core/fft.rs](../../sonara/src/core/fft.rs) | Прямые/обратные FFT, кеш планов | [DSP](dsp.md) |
| [core/spectrum.rs](../../sonara/src/core/spectrum.rs) | STFT/ISTFT, dB, PCEN, фазовый вокодер, Griffin–Lim и дополнительные спектральные преобразования | [DSP](dsp.md) |
| [core/constantq.rs](../../sonara/src/core/constantq.rs) | CQT/VQT и варианты обратного преобразования | [DSP](dsp.md) |
| [core/pitch.rs](../../sonara/src/core/pitch.rs) | YIN/pYIN, спектральные высоты и оценка строя | [DSP](dsp.md) |
| [core/harmonic.rs](../../sonara/src/core/harmonic.rs) | Интерполяция гармоник, salience и отслеживание гармоник заданного f0 | [DSP](dsp.md) |
| [core/convert.rs](../../sonara/src/core/convert.rs) | Частотные/временные шкалы, ноты и частотные весовые кривые | [DSP](dsp.md) |
| [core/notation.rs](../../sonara/src/core/notation.rs) | Музыкальная нотация, лады, svara и FJS | [DSP](dsp.md) |
| [core/intervals.rs](../../sonara/src/core/intervals.rs) | Интервальные отношения и частотные сетки | [DSP](dsp.md) |
| [dsp/mod.rs](../../sonara/src/dsp/mod.rs) | Группа базовых DSP-инструментов | [DSP](dsp.md) |
| [dsp/windows.rs](../../sonara/src/dsp/windows.rs) | Оконные функции для кадрового анализа | [DSP](dsp.md) |
| [dsp/iir.rs](../../sonara/src/dsp/iir.rs) | IIR, фильтрация вперёд/назад, каскады SOS | [DSP](dsp.md) |
| [dsp/extrema.rs](../../sonara/src/dsp/extrema.rs) | Локальные максимумы и минимумы | [DSP](dsp.md) |
| [filters.rs](../../sonara/src/filters.rs) | Банки фильтров mel/chroma и сопутствующие фильтры | [DSP](dsp.md) |
| [feature/mod.rs](../../sonara/src/feature/mod.rs) | Группа извлечения и обращения признаков | [DSP](dsp.md) |
| [feature/spectral.rs](../../sonara/src/feature/spectral.rs) | Mel/MFCC/chroma/tonnetz, спектральные статистики, RMS и ZCR | [DSP](dsp.md) |
| [feature/inverse.rs](../../sonara/src/feature/inverse.rs) | Приближённое восстановление спектра/аудио из mel/MFCC | [DSP](dsp.md#13-спектральные-признаки) |
| [feature/rhythm.rs](../../sonara/src/feature/rhythm.rs) | Темпограммы, metrogram и музыкальный размер | [Ритм](rhythm-tonality.md) |
| [onset.rs](../../sonara/src/onset.rs) | Сила атак, многополосные огибающие, выбор пиков | [Ритм](rhythm-tonality.md) |
| [beat.rs](../../sonara/src/beat.rs) | Кандидаты темпа, ACF, DP beat tracking, кривая темпа и PLP | [Ритм](rhythm-tonality.md) |
| [beatgrid.rs](../../sonara/src/beatgrid.rs) | Якорь сетки, начала тактов и устойчивость междолевых интервалов | [Ритм](rhythm-tonality.md) |
| [rhythmic_regularity.rs](../../sonara/src/rhythmic_regularity.rs) | Распределение атак внутри метрической сетки и отказ при недостатке данных | [Ритм](rhythm-tonality.md) |
| [tonal.rs](../../sonara/src/tonal.rs) | HPCP, распознавание аккордов и сенсорный диссонанс | [Гармония](rhythm-tonality.md) |
| [perceptual.rs](../../sonara/src/perceptual.rs) | LUFS, энергия, danceability, key/Camelot, valence, acousticness | [Признаки](descriptors-models.md) и [гармония](rhythm-tonality.md) |
| [loudness_ext.rs](../../sonara/src/loudness_ext.rs) | True peak, gain, short-term/momentary loudness, LRA | [Признаки](descriptors-models.md) |
| [mood.rs](../../sonara/src/mood.rs) | Приватная реализация эвристик настроения, экспорт через perceptual | [Признаки](descriptors-models.md) |
| [structure.rs](../../sonara/src/structure.rs) | Кривая энергии, novelty-сегментация, intro/outro | [Структура](descriptors-models.md) |
| [similarity.rs](../../sonara/src/similarity.rs) | 48-мерный вектор, профили весов, расстояние и similarity | [Сходство](descriptors-models.md) |
| [fingerprint.rs](../../sonara/src/fingerprint.rs) | Акустический отпечаток, сериализация и сопоставление записей | [Сходство](descriptors-models.md) |
| [vocal.rs](../../sonara/src/vocal.rs) | Историческая mel-эвристика вокальности, не текущий default конвейера | [Вокальность](descriptors-models.md) |
| [vocal_model.rs](../../sonara/src/vocal_model.rs) | Версионированная модель вокальности и встроенный артефакт | [Модели](descriptors-models.md) |
| [genre.rs](../../sonara/src/genre.rs) | JSON MLP, валидация/инференс пользовательских моделей, JSON-парсер | [Модели](descriptors-models.md) |
| [aggression.rs](../../sonara/src/aggression.rs) | Модель агрессивности, проверка артефактов и публичный API под feature | [Модели](descriptors-models.md) |
| [aggression_dsp.rs](../../sonara/src/aggression_dsp.rs) | Специализированное извлечение входов модели aggression | [Модели](descriptors-models.md) |
| [decompose.rs](../../sonara/src/decompose.rs) | HPSS, NMF, фильтрация по соседям | [DSP](dsp.md) |
| [effects.rs](../../sonara/src/effects.rs) | Изменение времени/высоты, тишина, remix и разделение мелодии | [DSP](dsp.md) |
| [segment.rs](../../sonara/src/segment.rs) | Матрицы повторяемости, cross-similarity и усиление путей | [DSP](dsp.md) |
| [sequence.rs](../../sonara/src/sequence.rs) | DTW, RQA, Viterbi, модели переходов | [DSP](dsp.md) |
| [util/mod.rs](../../sonara/src/util/mod.rs) | Экспорты общих утилит | [DSP](dsp.md) |
| [util/utils.rs](../../sonara/src/util/utils.rs) | Фреймирование, padding, нормализация, маски, работа с массивами | [DSP](dsp.md) |
| [util/matching.rs](../../sonara/src/util/matching.rs) | Сопоставление событий и интервалов | [DSP](dsp.md) |

## Модули Python-привязок

Файлы находятся в отдельном крейте; одинаковые имена с Rust-ядром не означают
повторную реализацию алгоритмов. Названия ниже — исходники, а не обещание
тождественной иерархии Python-импортов. Правила доступа описаны в [API](api.md).

| Исходник | Ответственность |
|---|---|
| [lib.rs](../../sonara-python/src/lib.rs) | Инициализация `_sonara`, версия, регистрация и верхнеуровневые экспорты |
| [analyze.rs](../../sonara-python/src/analyze.rs) | Анализ, конфигурация, result → dict, batch/progress, fingerprint match, augment |
| [aggression.rs](../../sonara-python/src/aggression.rs) | Legacy scorer и отдельные вызовы анализа aggression |
| [beat.rs](../../sonara-python/src/beat.rs) | Beat tracker, tempo curve и variability |
| [onset.rs](../../sonara-python/src/onset.rs) | Атаки, полосы и выбор метода onset strength |
| [tonal.rs](../../sonara-python/src/tonal.rs) | HPCP, аккорды, дескрипторы и диссонанс |
| [similarity.rs](../../sonara-python/src/similarity.rs) | Расстояние, similarity, константы версии и профилей |
| [effects.rs](../../sonara-python/src/effects.rs) | `trim`, `split`, `split_with_constraints`, `melody_separate` |
| [filters.rs](../../sonara-python/src/filters.rs) | Mel-банк |
| [error.rs](../../sonara-python/src/error.rs) | `SonaraError` → исключения Python и категории batch-ошибок |
| [util.rs](../../sonara-python/src/util.rs) | Регистрация пока пустого `util` |
| [core/mod.rs](../../sonara-python/src/core/mod.rs) | Регистрация функций audio/convert/spectrum в общем Python-модуле `core` |
| [core/audio.rs](../../sonara-python/src/core/audio.rs) | Загрузка, ресэмплинг, генераторы, LPC, autocorrelation, μ-law, блоки |
| [core/convert.rs](../../sonara-python/src/core/convert.rs) | Конвертации, нотация, интервалы и весовые кривые |
| [core/spectrum.rs](../../sonara-python/src/core/spectrum.rs) | STFT, CQT, pitch и гармонические операции |
| [feature/mod.rs](../../sonara-python/src/feature/mod.rs) | Регистрация общего модуля `feature` |
| [feature/spectral.rs](../../sonara-python/src/feature/spectral.rs) | Mel, MFCC, chroma, centroid и RMS |
| [feature/rhythm.rs](../../sonara-python/src/feature/rhythm.rs) | `tempo`, `metrogram`, `detect_time_signature` внутри `feature` |

## Чистый Python и артефакты

| Файл / каталог | Назначение |
|---|---|
| [__init__.py](../../python/sonara/__init__.py) | Фасад: результаты `TrackAnalysis`, bundled-path, обёртка similarity |
| [__init__.pyi](../../python/sonara/__init__.pyi) | Поддерживаемые вручную аннотации; runtime уточняется по привязкам |
| [_result.py](../../python/sonara/_result.py) | Наследник `dict`, `.failed`, `repr` и `.print()` |
| [display.py](../../python/sonara/display.py) | Matplotlib-отображение матриц/волны и форматирование осей |
| [genre.py](../../python/sonara/genre.py) | Обучение/сохранение/загрузка небольшого классификатора на NumPy |
| [vocal_model.py](../../python/sonara/vocal_model.py) | Обучение, калибровка и загрузка вокальной модели |
| [models](../../python/sonara/models/) | JSON-артефакты, включаемые в Python-пакет |
| [Rust models](../../sonara/models/) | Модель вокальности для встраивания в Rust |
| [aggression_linear.ferricml](../../sonara/src/aggression_linear.ferricml), [aggression_model.bin](../../sonara/src/aggression_model.bin) | Версионированные бинарные артефакты aggression, не аудиоданные |

## Зависимости и служебные части

[Workspace manifest](../../Cargo.toml) фиксирует численный стек: `ndarray`
для массивов, `rustfft`/`realfft` для FFT, `rayon` для параллелизма, `rubato`
для ресэмплинга, `symphonia = 0.6.1` и Hound для декодирования.
[vendor/hound-3.5.1](../../vendor/hound-3.5.1/) содержит исправление выравнивания
RIFF-чанков нечётной длины. Это внешняя библиотека с локальной поправкой,
а не ещё одно семейство MIR-алгоритмов SONARA.

[Тесты Rust](../../sonara/tests/) включают синтетическую проверку точности;
модульные тесты расположены рядом с реализацией. [Python-тесты](../../tests/)
проверяют привязки и контракты. [Примеры Rust](../../sonara/examples/) содержат
оценщики точности. [Benchmarks](../../sonara/benches/) измеряют скорость.
[Служебные скрипты](../../scripts/) управляют проверками API, fidelity и релизных
артефактов; [workflow](../../workflow/) описывает процедуры разработки.
Эти каталоги не входят в runtime API музыкального анализа.

## Границы интерпретации

Высокая `grid_stability` говорит о ровных интервалах между ударами, но не
исключает синкопированный рисунок. `instrumentalness` означает отсутствие
выраженного вокала, а не наличие живых инструментов. `similarity` ищет похожее
звучание, тогда как fingerprint сопоставляет одну запись в разных файлах.
`mood_aggressive` — эвристика настроения; `aggression_score` — другой,
обученный ранжирующий показатель. Значения `[0, 1]` не становятся
вероятностями только из-за своего диапазона.

Результаты зависят от содержимого, параметров и версии алгоритма. На тишине,
атональном материале и слабом ритме часть выводов не имеет музыкальной опоры.
Используйте confidence и возможность отказа, а не превращайте любую численную
оценку в безусловную метку. Точность на собственной коллекции проверяется
по ручной разметке; старые результаты benchmark не гарантируют её автоматически.

Нормативные правила хранения и совместимости находятся в
[consumer-contract.md](../consumer-contract.md). Этот русский справочник
поясняет их, но не создаёт второй независимый контракт.
