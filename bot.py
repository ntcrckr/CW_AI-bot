from aiogram import Bot, Dispatcher, executor, types
import transformers_model
import lstm_model
from config import TOKEN

API_TOKEN = TOKEN

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.answer("""\
Привет!
Я бот для анализа текста отзыва.
Пришли мне текст и я предположу его эмоциональную оценку!\
""")


@dp.message_handler()
async def analysis(message: types.Message):
    msg = await message.answer("""\
Подожди, пока модели дадут ответ.\
""")
    # call the models
    lstm_prediction = lstm_model.get_predictions(message.text)
    bert_prediction = transformers_model.Model().get_prediction(message.text)
    # call the models
    await msg.delete()
    await message.reply(f"""\
Модели дали следующие ответы:
  - LSTM: {lstm_prediction} ⭐
  - transformers Bert: {bert_prediction} ⭐\
""")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
