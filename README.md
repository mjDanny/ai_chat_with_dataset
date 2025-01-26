```markdown
# 🤖 Fine-Tuning DistilBERT для задачи Question Answering (QA)

Этот проект демонстрирует процесс тонкой настройки модели **DistilBERT** для задачи **Question Answering (QA)** с использованием датасета **SQuAD**. Мы используем библиотеки **Hugging Face Transformers** и **TensorFlow** для обучения модели и её последующего использования для ответов на вопросы.

## 📋 Оглавление
1. [Описание проекта](#описание-проекта)
2. [Установка и настройка](#установка-и-настройка)
3. [Использование](#использование)
4. [Примеры работы модели](#примеры-работы-модели)
5. [Автор](#автор)
6. [Лицензия](#лицензия)

---

## 🚀 Описание проекта

В этом проекте мы:
- Загружаем и подготавливаем датасет **SQuAD**.
- Токенизируем данные с помощью **DistilBERT**.
- Обучаем модель на подмножестве данных.
- Сохраняем обученную модель в **Hugging Face Hub**.
- Используем модель для ответов на вопросы на основе предоставленного контекста.

Модель **DistilBERT** была выбрана из-за её эффективности и меньшего размера по сравнению с BERT, что позволяет быстрее обучать модель без значительной потери качества.

---

## ⚙️ Установка и настройка

Для запуска проекта необходимо установить следующие зависимости:

```bash
pip install transformers datasets evaluate tensorflow
```

### Настройка Hugging Face Hub
Для загрузки и сохранения модели в Hugging Face Hub, необходимо авторизоваться:

```python
from huggingface_hub import login

HF_TOKEN = "ваш_токен"
login(HF_TOKEN, add_to_git_credential=True)
```

---

## 🛠 Использование

### 1. Загрузка и подготовка данных
Датасет SQuAD загружается и разбивается на обучающую и валидационную выборки:

```python
from datasets import load_dataset

dataset = load_dataset("squad", split="train[:5000]")
dataset = dataset.train_test_split(test_size=0.2)
```

### 2. Токенизация данных
Используется токенизатор **DistilBERT** для обработки текста:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

### 3. Обучение модели
Модель обучается с использованием **TensorFlow**:

```python
from transformers import TFAutoModelForQuestionAnswering

model = TFAutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
model.compile(optimizer=optimizer)
model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=5, callbacks=[callback])
```

### 4. Сохранение модели
После обучения модель сохраняется в Hugging Face Hub:

```python
callback = PushToHubCallback(
    output_dir="my_qa_model",
    tokenizer=tokenizer,
)
```

---

## 🧠 Примеры работы модели

### Пример 1: Дата выпуска первого iPhone
**Вопрос:**  
*Когда был выпущен первый iPhone?*  
**Контекст:**  
*Первый iPhone был выпущен 29 июня 2007 года. Это был революционный продукт, который изменил индустрию смартфонов.*  
**Ответ модели:**  
*"29 июня 2007"*

### Пример 2: Автор романа "Убить пересмешника"
**Вопрос:**  
*Кто написал роман "Убить пересмешника"?*  
**Контекст:**  
*Роман "Убить пересмешника" был написан Харпер Ли и опубликован в 1960 году.*  
**Ответ модели:**  
*"Харпер Ли"*

### Пример 3: Столица Франции
**Вопрос:**  
*Какая столица Франции?*  
**Контекст:**  
*Столица Франции — Париж. Он известен своим искусством, культурой, кухней и модой.*  
**Ответ модели:**  
*"Париж"*

---
