from transformers import AutoModelForQuestionAnswering,AutoTokenizer,pipeline
import streamlit as st

model = AutoModelForQuestionAnswering.from_pretrained('uer/roberta-base-chinese-extractive-qa')
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-chinese-extractive-qa')
QA = pipeline('question-answering', tokenizer=tokenizer, model=model)

st.title('中文問答系統')

defalt_q = '著名诗歌《假如生活欺骗了你》的作者是'
question = st.text_input('輸入你的問題', value = defalt_q)

defalt_c = '''普希金从那里学习人民的语言，吸取了许多有益的养料，这一切对普希金后来的创作产生了很大的影响。这两年里，普希金创作了不少优秀的作品，如《囚徒》、《致大海》、《致凯恩》和《假如生活欺骗了你》等几十首抒情诗，叙事诗《努林伯爵》，历史剧《鲍里斯·戈都诺夫》，以及《叶甫盖尼·奥涅金》前六章。'''

context = st.text_area("輸入可以找到問題的文本", value = defalt_c, height = 150)

QA_input = {'question': question, 'context': context}

if st.button('尋找答案'):
    answer = QA(QA_input)
    st.success(f"Answer: {answer['answer']} (Score: {round(answer['score'],2)})")