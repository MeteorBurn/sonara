# API, конвейер анализа и хранение результатов

[К оглавлению](README.md) · [DSP](dsp.md) · [Ритм и гармония](rhythm-tonality.md) · [Признаки и модели](descriptors-models.md)

SONARA предлагает два уровня работы. `analyze_*` превращает запись в готовый
набор признаков, переиспользуя вычисления. Отдельные DSP-функции дают доступ к
спектрограммам, высоте тона и другим промежуточным представлениям. Второй путь
удобен для экспериментов, но последовательные вызовы с `y=` могут повторять FFT.

Источники: [Rust-конвейер](../../sonara/src/analyze.rs),
[Python-привязки](../../sonara-python/src/analyze.rs),
[Python-фасад](../../python/sonara/__init__.py),
[контракт потребителя](../consumer-contract.md).

## Версия и установка

Эта документация описывает исходники **0.3.7**, ревизию `0c2a4b3`.
Версия пакета, версия схемы результата и версия embedding — разные величины.
Для этого состояния `ANALYSIS_SCHEMA_VERSION = 6`, `SIMILARITY_VERSION = 2`.
Установленный пакет проверяется через `sonara.__version__`.

Python-пакету нужны Python 3.10+ и NumPy `>=1.23,<3`; служебным скриптам
репозитория — Python 3.11+. Для сборки расширения нужны Rust и Maturin.
Установка `sonara` с PyPI выбирает upstream, а не обязательно этот форк.
Старый wheel `v0.3.6-meteorburn.1` не следует считать сборкой исходников 0.3.7.
Доступность новых опубликованных пакетов нужно проверять отдельно: данный
справочник не является каталогом релизов.

Обычная установка из исходников в отдельное окружение, PowerShell,
из корня нужной копии репозитория:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install .
& .\.venv\Scripts\python.exe -c "import sonara; print(sonara.__version__)"
```

Команда установки собирает расширение и может загружать зависимости.
Для рабочего окружения разработчика с уже установленным Maturin применяют
`maturin develop --release -m sonara-python/Cargo.toml` в активном virtualenv.
Rust-потребитель подключает крейт `sonara`; для локального checkout пример
зависимости в его собственном manifest — `sonara = { path = "../sonara/sonara" }`.
Путь нужно соотнести с расположением проекта потребителя.
Если подключаете форк из другого Cargo workspace, также перенесите
`[patch.crates-io]` для vendored Hound в корневой manifest потребителя:
Cargo не наследует patch-секцию зависимости автоматически.

## Минимальный пример без аудиофайла

Синусоида позволяет проверить установку и форму результата, но не точность
музыкального анализа: у непрерывного тона нет размеченного музыкального темпа.

```python
import numpy as np
import sonara

sr = 22050
t = np.arange(sr * 3, dtype=np.float32) / sr
y = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

result = sonara.analyze_signal(y, sr=sr, mode="playlist")
assert isinstance(result, dict)
assert abs(result["duration_sec"] - 3.0) < 1e-5
assert len(result["mfcc_mean"]) == 13
assert len(result["chroma_mean"]) == 12
print(result["provenance"])
```

`analyze_signal` ожидает одномерный NumPy-массив `float32`. `sr` здесь означает
**фактическую** частоту дискретизации массива, а не команду пересэмплировать его.
Для изменения частоты есть `sonara.resample(y, orig_sr=..., target_sr=...)`.
Пустой массив, `sr=0`, NaN и бесконечности основной анализатор отвергает.

## Точки входа Python и Rust

| Задача | Python | Rust, пространство `sonara::analyze` |
|---|---|---|
| Файл | `analyze_file(path, *, sr=22050, mode="compact", features=None, ...)` | `analyze_file(&Path, u32, &AnalysisConfig) -> Result<TrackAnalysis>` |
| Массив | `analyze_signal(y, *, sr=22050, mode="compact", features=None, ...)` | `analyze_signal(ArrayView1<Float>, u32, &AnalysisConfig)` |
| Набор файлов | `analyze_batch(paths, ..., progress=None)` | `analyze_batch(&[&Path], u32, &AnalysisConfig) -> Vec<Result<TrackAnalysis>>` |
| Прогресс | `progress(done, total)` | `analyze_batch_with(..., on_done)` |
| Дозаполнение | `augment_analysis(cached, features, *, audio_path=None, ...)` | `augment_analysis(&cached, &[&str], Option<&Path>, &config)` |
| Проверка дозаполнения | `can_augment(cached, name)`; `augment_blocker(cached, name)` | Одноимённые функции; типизированный `AugmentBlocker` |
| Зависимости признаков | `feature_dependencies()` | `feature_dependencies()`; `feature_dependency(name)` |

Параметры общего анализатора также включают `bpm_min`, `bpm_max`, `genre_model`,
`vocalness_model`. Python принимает пути к моделям; Rust — `Option<Arc<...>>`.
`vocalness_model="bundled"` в Python выбирает модель из пакета. По умолчанию
этот аргумент равен `None`: сам факт наличия модели в поставке её не включает.
Готовой жанровой модели нет.

На файловом пути `sr=0` сохраняет исходную частоту; при положительном `sr`
загрузчик приводит аудио к указанной частоте и моно. У массива нет контейнерных
тегов, поэтому `analyze_signal` не извлекает `tags`.

Минимальный файловый вызов Rust, без PyO3 и Python:

```rust
use std::path::Path;
use sonara::analyze::{analyze_file, AnalysisConfig, AnalysisMode};

fn main() -> sonara::Result<()> {
    let config = AnalysisConfig {
        mode: AnalysisMode::Playlist,
        bpm_min: Some(79.0),
        bpm_max: Some(192.0),
        ..Default::default()
    };
    let result = analyze_file(Path::new("track.mp3"), 22050, &config)?;
    println!("BPM: {}, beats: {:?}", result.bpm, result.beats_sec());
    Ok(())
}
```

Это программа потребителя библиотеки, а не встроенная команда SONARA.
Конструкторы `compact()`, `playlist()`, `full()` возвращают `AnalysisConfig`.
Список Rust-признаков — `Option<HashSet<String>>`; например:
`Some(["embedding".to_owned()].into_iter().collect())`.

## Режимы и явный выбор признаков

| Режим | Что добавляется |
|---|---|
| `compact` | Базовые BPM/удары/атаки, RMS, LUFS, динамический диапазон, центроид, ZCR, длительность |
| `playlist` | Спектральные признаки, MFCC, chroma, аккорды, диссонанс, тональность и перцептивные эвристики |
| `full` | К `playlist` добавляются кривая темпа и оценка размера |

**`features` заменяет выбор режима, а не объединяется с ним.** Поэтому
`mode="playlist", features=["beatgrid"]` — не «playlist плюс сетка».
При этом базовые поля вычисляются всегда, а зависимости отдельных групп могут
добавлять вычисления и поля. Явный список не является строгой проекцией словаря
на перечисленные ключи: например, запрос расширенного признака вычисляет
общий спектральный блок, а `embedding` публикует его компоненты.

Если нужны все штатные признаки `playlist` и дополнительные группы, набор
можно получить из реестра, не копируя вручную его состав:

```python
import sonara

playlist_features = [
    row["name"] for row in sonara.feature_dependencies()
    if not row["opt_in_only"] and not row["full_only"]
]
features = playlist_features + ["beatgrid", "onset_bands", "rhythmic_regularity"]
result = sonara.analyze_file("track.mp3", features=features)
```

Ни один режим, включая `full`, автоматически не включает opt-in группы.
Имена регистронезависимы; неизвестное имя является ошибкой. `genre` не является
именем группы — запросом классификации служит параметр `genre_model`.

### Полный реестр групп

`C` — нужны покадровые данные и повторное чтение аудио при augment;
`A` — нужен сигнал или контейнер; `S` — можно вычислить из сохранённых полей,
если они есть; `E` — сборка embedding из сохранённых признаков.

| Имя в `features` | Основные поля результата | Доступность | Класс |
|---|---|---|---|
| `bpm` | `bpm`, `bpm_raw`, `bpm_confidence`, `bpm_candidates` | База | C |
| `beats` | `beats`; в Python также `n_beats` | База | C |
| `onsets` | `onset_frames` | База | C |
| `rms` | `rms_mean`, `rms_max` | База | C |
| `dynamic_range` | `dynamic_range_db` | База | C |
| `centroid` | `spectral_centroid_mean` | База | C |
| `zcr` | `zero_crossing_rate` | База | A |
| `onset_density` | `onset_density` | База | S |
| `bandwidth` | `spectral_bandwidth_mean` | Playlist | C |
| `rolloff` | `spectral_rolloff_mean` | Playlist | C |
| `flatness` | `spectral_flatness_mean` | Playlist | C |
| `contrast` | `spectral_contrast_mean` | Playlist | C |
| `mfcc` | `mfcc_mean` | Playlist | C |
| `chroma` | `chroma_mean` | Playlist | C |
| `chords` | `chord_sequence`, `chord_events`, `predominant_chord`, `chord_change_rate` | Playlist | C |
| `dissonance` | `dissonance` | Playlist | C |
| `energy` | `energy` | Playlist | S |
| `danceability` | `danceability` | Playlist | S |
| `key` | `key`, `key_confidence`, `key_camelot` | Playlist | S |
| `valence` | `valence` | Playlist | S |
| `acousticness` | `acousticness` | Playlist | S |
| `tempo_curve` | `tempo_curve`, `tempo_variability` | Full | S |
| `time_signature` | `time_signature`, `time_signature_confidence` | Full | C |
| `onset_bands` | `onset_strength_bands`, `onset_band_edges_hz` | Opt-in | C |
| `beatgrid` | `grid_offset_sec`, `downbeats`, `grid_stability` | Opt-in | C |
| `rhythmic_regularity` | Оценка, метка, confidence, candidates с префиксом `rhythmic_regularity` | Opt-in | C |
| `structure` | `energy_curve`, `energy_curve_hop_sec`, `segments`, `intro_end_sec`, `outro_start_sec`, `energy_level` | Opt-in | C |
| `embedding` | `embedding`, `embedding_version` и компоненты | Opt-in | E |
| `aggression` | `aggression_score`, `aggression_confidence`, `aggression_forcefulness`, `aggression_harshness`, `aggression_tension`, `aggression_rhythm` | Opt-in, Cargo feature | A |
| `fingerprint` | `fingerprint`; Python добавляет `fingerprint_version` | Opt-in | A |
| `loudness` | `true_peak_db`, `replaygain_db`, `loudness_curve`, `loudness_momentary_max_db`, `loudness_range_lu` | Opt-in | A |
| `silence` | `leading_silence_sec`, `trailing_silence_sec` | Opt-in | C |
| `key_candidates` | `key_candidates` | Opt-in | S |
| `vocalness` | `vocalness` | Opt-in или модель | S* |
| `mood` | `mood_happy`, `mood_aggressive`, `mood_relaxed`, `mood_sad` | Opt-in | S |
| `instrumentalness` | `instrumentalness` | Opt-in или модель | S* |
| `tags` | Словарь `tags` | Opt-in, файловый вход | A |

`duration_sec`, `provenance` и `loudness_lufs` присутствуют в базовом результате
без отдельного имени группы. `S*` описывает встроенную эвристику: модель
вокальности вместо этих входов требует признаки для embedding. Статическая
карта зависимостей сама по себе не доказывает возможность конкретного augment.

## Как устроен объединённый проход

После декодирования проверяются вход и конфигурация. Основной анализ использует
окно Hann на **2048 отсчётов**, шаг **512**, **128 mel-полос**. Сигнал дополняется
нулями слева и справа на половину окна. На каждом кадре выполняется real FFT;
его модуль и мощность используются несколькими признаками сразу.

Из одного спектра получаются mel-энергии, спектральный центроид, а в расширенном
проходе — bandwidth, rolloff, flatness, chroma, contrast, HPCP и диссонанс.
RMS считается по временному кадру **без** умножения на Hann. Далее log-mel даёт
onset envelope, алгоритм темпа и DP-трекер находят удары; DCT-II даёт MFCC;
агрегаты питают перцептивные признаки и модели. Полный массив линейной
спектрограммы мощности не сохраняется, но mel-матрица и нужные покадровые
представления в памяти остаются. Это не анализ произвольного длинного файла
с константной памятью.

При количестве кадров от 32 используется параллельная обработка Rayon.
Банк фильтров, DCT и FFT-планы переиспользуются через кеши. Batch также
распараллелен по файлам. Это объясняет устройство оптимизации, но не обещает
фиксированное время на трек: декодирование, диск, длительность, частота и набор
признаков существенно влияют на результат.

«Один проход» относится к общей спектральной работе, а не ко всем операциям
вообще: LUFS и true peak работают во временной области, fingerprint имеет своё
представление. Aggression использует канонические 22050 Гц; при другой частоте
основного анализа создаётся отдельная каноническая ветка. На файловом пути
обе ветки получают данные из одного декодирования исходника.

## Единицы, формы и временная ось

| Данные | Представление и смысл |
|---|---|
| `beats`, `onset_frames`, `downbeats` | Индексы кадров, **не секунды и не отсчёты аудио** |
| `duration_sec`, границы событий/тишины | Секунды |
| `bpm`, `bpm_raw`, `tempo_curve` | Удары в минуту |
| `onset_density`, `chord_change_rate` | События в секунду |
| Центроид, bandwidth, rolloff, края onset-полос | Гц |
| `rms_mean`, `rms_max` | Линейная амплитуда RMS, не dB и не LUFS |
| `zero_crossing_rate` | Доля смен знака между соседними отсчётами в основном анализаторе |
| `loudness_lufs`, `loudness_curve` | LUFS |
| `true_peak_db` | dBTP |
| `replaygain_db`, `dynamic_range_db` | dB; разные физические величины |
| `loudness_range_lu` | LU |
| `mfcc_mean`, `chroma_mean`, `spectral_contrast_mean` | Списки длиной 13, 12 и 7 в общем анализаторе |
| `embedding` | 48 чисел; обязательно хранить `embedding_version` |
| `onset_strength_bands` | Список полос, внутри каждой список кадров |
| `fingerprint` | Python: base64-строка; Rust: `Vec<u32>` |

Для основного прохода время кадра `k` равно `k * hop_length / sample_rate`.
При 22050 Гц один шаг — примерно 23,22 мс, а длина окна — 92,88 мс. Малый
шаг не означает такое же независимое временное разрешение: соседние окна
перекрываются.

```python
# result — успешный результат analyze_*.
prov = result["provenance"]
beat_seconds = [
    frame * prov["hop_length"] / prov["sample_rate"]
    for frame in result["beats"]
]
```

Кривая энергии имеет собственный `energy_curve_hop_sec`, громкость — окна 3 с
с шагом 1 с, а `tempo_curve[i]` относится к интервалу между `beats[i]` и
`beats[i+1]`. Эти массивы нельзя соединять простым совпадением индексов.
`chord_events` и `segments` уже содержат `start_sec`/`end_sec`; события аккордов
объединяют соседние одинаковые метки. `N` — отсутствие определённого аккорда.

В Rust необязательные поля имеют `Option<T>`. В Python обычное отсутствующее
поле чаще **не включается** в словарь. Исключение — измерения с отказом от
оценки: при запрошенной rhythmic regularity или aggression confidence
сохраняется, а сама оценка может быть `None`.

## BPM и диапазон темпа

`bpm_raw` сохраняет выбранный темп до октавного приведения; `bpm` — после.
Параметры диапазона задаются только вместе: конечные положительные числа,
`bpm_min < bpm_max` и `bpm_max >= 2 * bpm_min`. Например, 64 BPM удваивается
до 128 в диапазоне 79–192. Это не произвольное ограничение результата до
ближайшей границы и не доказательство, что музыкально правильна именно эта
октава. Оба значения диапазона сохраняются в provenance.

## Пакетная обработка и ошибки

```python
import sonara

paths = ["track-a.mp3", "track-b.flac"]
results = sonara.analyze_batch(paths, mode="playlist")
for item in results:
    if item.failed:
        print(item["path"], item["error_kind"], item["error"])
        continue
    print(item["path"], item["bpm"], item.get("key"))
```

Порядок результатов совпадает с порядком входных путей; каждая запись batch
содержит `path`. Ошибка отдельного файла даёт запись с `error` и `error_kind`,
не отменяя остальные. Ошибки настройки всего вызова, например неизвестный
режим или невозможность загрузить модель, могут поднять исключение до обхода.

Категории из [адаптера ошибок](../../sonara-python/src/error.rs): `io`, `decode`,
`unsupported_format`, `invalid_audio`, `model`, `insufficient_data`, `compute`.
В одиночном вызове некорректные параметры/форма/модель обычно дают `ValueError`,
ошибка файла/декодера — `IOError`, неподдержанный формат — `NotImplementedError`,
численная ошибка — `RuntimeError`. Ошибки преобразования аргументов PyO3 могут
возникнуть раньше и иметь другой тип, например `TypeError` для неверного dtype.

`progress(done, total)` вызывается после завершения каждого файла, включая
неуспешный; это порядок завершения, а не индекс входного списка. Исключение
в callback не отменяет batch. Callback должен быть коротким. В текущей
привязке GIL освобождается вокруг batch **в ветке с callback**; ветка
`progress=None` напрямую вызывает Rust. Это не отменяет внутренний Rayon,
но важно, если другие Python-потоки должны продолжать работу.

## Кеш, JSON и дозаполнение

`TrackAnalysis` в Python — наследник `dict`, не NumPy-матрица и не ORM-модель.
Его можно сериализовать стандартным `json`. После чтения JSON достаточно
обычного словаря исходной формы; `.print()` и `.failed` доступны у класса,
но не у обычного результата `json.loads`.

```python
import json
import sonara

base = sonara.analyze_file("track.mp3", mode="playlist")
cached = json.loads(json.dumps(base))

assert sonara.can_augment(cached, "mood")
with_mood = sonara.augment_analysis(cached, ["mood"])
assert "mood_happy" not in cached  # исходный словарь не меняется

updated = sonara.augment_analysis(
    with_mood,
    ["onset_bands", "rhythmic_regularity"],
    audio_path="track.mp3",
)
```

`augment_analysis` возвращает копию с запрошенными полями, не очищая остальные.
Отсутствие исходных скаляров, класс `Audio`/`FrameCurves` или несовместимые версии
объясняются через `augment_blocker`. `can_augment=False` означает «не получится
без аудио», если причиной является `NeedsAudio`, а не запрет на дозаполнение.
Проверка `can_augment` для вокальности относится к встроенной эвристике:
требования явно переданной модели проверяются внутри самого augment.

Для `onset_bands` и `rhythmic_regularity` в 0.3.7 аудио требуется всегда,
даже когда полосы уже сохранены. Повторный анализ использует частоту из
provenance и записанный BPM-диапазон; аргументы диапазона в augment служат
резервом для старых записей без этих метаданных. Обновление вокальности или
инструментальности обновляет **обе** величины и `vocalness_model_id`.

Для возобновляемого расчёта различайте состояния «нет поля» и «измерено, но
нет надёжной оценки». Присутствие `rhythmic_regularity_confidence`, даже `0.0`,
отмечает выполненное измерение. Значение `None` у score — не повод бесконечно
ставить трек обратно в очередь. Массивы полос сохраняйте вместе с границами.

Добавление нового поля не меняет версию всей схемы. Изменение смысла или единиц
существующего поля меняет её. Для кеша дополнительно важны версия embedding,
версия fingerprint, идентификаторы моделей и параметры анализа. У именованных
профилей similarity собственные версии весов; смена профиля не изменяет
сохранённый невзвешенный embedding.

SONARA не создаёт таблицы, не мигрирует базу и не решает, как отождествлять
файлы. Не передавайте произвольную строку своей БД вместо полного результата
SONARA; используйте адаптер или целевой `analyze_file` с переносом только новых
полей. При несовместимой схеме результата нужен новый анализ, а не смешивание
полей разных поколений.

## Отдельные DSP-вызовы

Пример общего спектра для нескольких дальнейших операций:

```python
import numpy as np
import sonara

y = sonara.tone(440.0, sr=22050, length=22050)
S = sonara.stft(y, n_fft=2048, hop_length=512)
power = np.abs(S).astype(np.float32) ** 2
mel = sonara.melspectrogram(S=power, sr=22050.0, n_fft=2048)
mel_db = sonara.power_to_db(mel)
coefficients = sonara.mfcc(S=mel_db, sr=22050.0, n_mfcc=13)
assert coefficients.shape[0] == 13
```

Здесь `melspectrogram(S=...)` получает уже рассчитанный спектр, а `mfcc(S=...)`
ожидает **log-mel**, не комплексный STFT. У standalone MFCC по умолчанию 20
коэффициентов, у объединённого анализатора — 13. Матрицы спектральных признаков
обычно имеют форму `(частоты или признаки, кадры)`; STFT комплексный, мощность
вещественная. Специфику других преобразований см. в [DSP](dsp.md).

### Python не является полным зеркалом Rust

Фактический экспорт задают [регистрация расширения](../../sonara-python/src/lib.rs)
и `register` каждого binding-модуля, а не перечень всех алгоритмов Rust.
Например, `sonara.feature.tempo` и `sonara.feature.metrogram` зарегистрированы
в `feature`, но не продублированы как `sonara.tempo`/`sonara.metrogram`.
Спектральные привязки также живут непосредственно в `feature`;
`feature.spectral` не создаётся отдельным Python-подмодулем.
Аналогично функции `core` находятся непосредственно в `sonara.core`:
`sonara.core.load`, `sonara.core.stft`; вложенных `core.audio` и `core.spectrum`
в runtime нет, хотя исходники привязок разложены по таким файлам.
`sonara.util` сейчас пустой контейнер, не зеркало Rust `util`.

HPSS/NMF, DTW/Viterbi, часть эффектов, фильтров и спектральных функций доступны
в Rust, но не имеют соответствующих привязок в этой версии. Наличие имени
в [typing-stub](../../python/sonara/__init__.pyi) или старом перечне README
не гарантирует наличие функции в runtime. У преобразований учитывайте
регистр: например, `A_weighting`, `A4_to_tuning`, `tuning_to_A4`.

### Отображение

[display.py](../../python/sonara/display.py) предоставляет `specshow`, `waveshow`,
`cmap` и форматирование времени/нот/chroma/tonnetz через Matplotlib.
Matplotlib — необязательная зависимость: при её отсутствии импорт пакета
допускается, но рисование завершается `ImportError`.
`specshow` ожидает двумерные данные; по умолчанию mel-ось строится по индексам
полос. Для физических частотных координат передавайте `y_coords` явно.
`waveshow` строит обычную линию всех отсчётов, а не многоуровневую огибающую
для произвольно больших файлов.

## Проверка и развитие проекта

У [ядра](../../sonara/Cargo.toml) пустой набор Cargo features по умолчанию.
`aggression` включает FerricML и SHA-256; Python-привязка включает этот feature.
`accelerate` включает Apple Accelerate/BLAS; `bench-internals` нужен для
внутреннего benchmark модели и подразумевает aggression. PyO3 и NumPy остаются
в binding-крейте, а не попадают в Rust-ядро.

Команды проверки из [руководства участника](../../CONTRIBUTING.md):

```powershell
cargo test -p sonara
cargo test -p sonara --features aggression
cargo check -p sonara-python
python scripts/run_python_tests.py --check-contract
python scripts/run_python_tests.py
```

Последняя команда требует собранного Python-расширения. Прямой `pytest` не
заменяет штатный runner: часть тестов — исполняемые скрипты с собственным
драйвером. Изменения BPM/тональности/аккордов требуют чисел до и после на
размеченных данных; benchmark скорости не подтверждает точность. Маршрутизация
таких проверок описана в [fidelity gates](../../tests/fidelity_gates.json).
Эта страница описывает команды, а не заявляет, что все они запускались при
подготовке документации.
