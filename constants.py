PROMPTS = [
    'What kind of chart is this? Choose between these chart types: "Column, Bar, Line, Pie, Table". Give a one word answer!',
    'What is the title of the chart?',
    'What are the names of the varibles or columns in this chart? Give the answer in a list.',
    'What is the time period shown in this chart?',
    'Caption this chart.']

MODEL = "llama3.2-vision"

CSV_FILENAME = "analysis_results.csv"

FIELDNAMES = ['Image ID', PROMPTS[0], PROMPTS[1], PROMPTS[2], PROMPTS[3]]

SYSTEM_PROMPT = "You are an expert in reading graphs and understanding the data from it."
