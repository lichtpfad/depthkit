# Установка depthkit в TouchDesigner

TouchDesigner использует собственную сборку Python (3.11.10, Derivative) — в неё нельзя просто сделать `pip install`. Ниже два рабочих способа.

---

## Способ 1 — Python Module Path (рекомендуется)

Указываем TD путь к site-packages нашего venv. TD добавит его в `sys.path` при старте.

### Шаги

**1. Убедиться, что venv создан**

```cmd
cd C:\work\depthkit
uv venv .venv --python 3.11
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install transformers accelerate
uv pip install -e .
```

> `uv` = `C:\Users\stani\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe`

**2. Прописать путь в TD**

`Edit → Preferences → Python → Python 64-bit Module Path`

Добавить (через точку с запятой если уже что-то есть):

```
C:\work\depthkit\.venv\Lib\site-packages
```

**3. Перезапустить TouchDesigner**

**4. Проверить в Textport (`Alt+T`)**

```python
import depthkit
print(depthkit.__version__)  # 0.1.0

import torch
print(torch.cuda.is_available())  # True
```

---

## Способ 2 — Execute DAT (без перезапуска TD)

Полезно для первой установки или если нет доступа к Preferences.

**1. Создать Text DAT, назвать `install_depthkit`**

```python
# Вставить в Text DAT:
import subprocess, sys

# Путь к pip нашего venv
pip = r"C:\work\depthkit\.venv\Scripts\pip.exe"

packages = [
    "torch torchvision --index-url https://download.pytorch.org/whl/cu124",
    "transformers accelerate",
    "depthkit",  # или "-e C:\\work\\depthkit" для editable install
]

for pkg in packages:
    subprocess.run([pip, "install"] + pkg.split(), check=True)

print("Done. Restart TD or add site-packages to Python Module Path.")
```

**2. Создать Execute DAT, подключить к Text DAT, нажать Execute**

Дождаться завершения (видно в Textport).

**3. Добавить путь в Preferences** (как в Способе 1, шаг 2) и перезапустить TD.

---

## Использование в Script TOP

После установки — создать Script TOP, вставить код:

```python
import sys
# На случай если Module Path ещё не прописан:
sys.path.insert(0, r"C:\work\depthkit\.venv\Lib\site-packages")
sys.path.insert(0, r"C:\work\depthkit")

from depthkit.drivers.td import DepthTD

_depth_td = None

def onSetupParameters(scriptOp):
    pass

def onCook(scriptOp):
    global _depth_td
    if _depth_td is None:
        _depth_td = DepthTD(model="vitb")   # vits / vitb / vitl
        _depth_td.warmup()                  # загружает веса (первый раз ~3 сек)

    if len(scriptOp.inputs) == 0:
        return

    arr = _depth_td.cook_numpy(scriptOp.inputs[0])  # (H, W, 1) float32, [0..1]
    scriptOp.copyNumpyArray(arr)
```

Подать на вход любой TOP (Webcam, Video File In, Composite и т.д.).

Выход — одноканальная текстура глубины. Подключить **Level TOP** с `Gamma = 0.4545` если нужна визуализация в sRGB.

---

## Параметры DepthTD

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `model` | `"vitb"` | `"vits"` (быстрее), `"vitb"`, `"vitl"` (точнее) |
| `max_res` | `640` | Длинная сторона при инференсе. Уменьшить для скорости |
| `cache_dir` | `None` | Папка для кэша HuggingFace моделей |

Первый `warmup()` скачивает модель (~400 MB для vitb) если её нет в кэше HuggingFace (`~/.cache/huggingface`).

---

## Диагностика

**`ModuleNotFoundError: No module named 'depthkit'`**
→ Module Path не прописан или TD не перезапущен после изменения.

**`No module named 'torch'`**
→ `torch` не установлен в venv. Запустить `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`

**`CUDA не доступна` / `torch.cuda.is_available() = False`**
→ Установлена CPU-версия torch. Переустановить с `--index-url https://download.pytorch.org/whl/cu124`

**Script TOP зависает на первом куке**
→ Модель загружается — подождать 3-5 секунд. После `warmup()` — ≤30 ms/кадр.

**Низкий FPS (<15)**
→ Использовать `model="vits"` и `max_res=512`

---

## IPC-сервер (альтернатива Script TOP)

Если хочется изолировать Python-процесс от TD:

```cmd
cd C:\work\depthkit
.venv\Scripts\python.exe -m depthkit.drivers.td --model vitb
```

Требует cuda-link (forkni/cuda-link) с обеих сторон: CUDAIPCExporter в TD → сервер → CUDAIPCImporter в TD.
