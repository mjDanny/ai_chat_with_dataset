# Модель для ответа на вопросы (Question Answering)

## Описание

Проект реализует модель для автоматического ответа на вопросы на основе контекста, используя библиотеку Hugging Face и TensorFlow. В основе лежит предобученная модель DistilBERT, дообученная на датасете SQuAD.

---

## Функционал

- Загрузка и предобработка данных SQuAD.
- Токенизация вопросов и контекстов.
- Обучение модели для извлечения ответов.
- Сохранение модели в Hugging Face Hub.
- Использование модели для предсказания ответов на новые вопросы.

---

## Используемые технологии

- **Python**
- **Hugging Face Transformers**
- **TensorFlow**
- **Datasets**

---

## Установка и запуск

1. Установите зависимости:
   ```bash
   pip install transformers tensorflow datasets huggingface_hub

2. Авторизуйтесь в Hugging Face:

```from huggingface_hub import login
   login("ВАШ_ТОКЕН")
```
3.Запустите скрипт для обучения и предсказаний:

    python main.py

Пример использования

После обучения можно задать вопрос и получить ответ:

```from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

question = "Когда был выпущен первый iPhone?"
context = "Первый iPhone был выпущен 29 июня 2007 года."

tokenizer = AutoTokenizer.from_pretrained("my_qa_model")
model = TFAutoModelForQuestionAnswering.from_pretrained("my_qa_model")

inputs = tokenizer(question, context, return_tensors="tf")
outputs = model(**inputs)

answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
answer = tokenizer.decode(predict_answer_tokens)

print("Ответ:", answer)  # Ответ: "29 июня 2007 года"
```
