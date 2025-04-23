import pandas as pd
from datetime import datetime
from langchain_community.llms import Ollama
from langchain.output_parsers.json import parse_json_markdown
import smtplib
from email.message import EmailMessage


data = {
    "month": [
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December"
    ],
    "weather": [
        "Winter/Rainy", "Winter/Rainy", "Rainy", "Rainy", "Summer", "Summer",
        "Summer", "Summer", "Summer", "Winter", "Winter", "Winter/Rainy"
    ]
}
df = pd.DataFrame(data)

current_month = datetime.now().strftime("%B")
month_index = df[df['month'] == current_month].index[0]
order_ahead_months = [(month_index + i) % 12 for i in range(4, 7)]
next_season_df = df.iloc[order_ahead_months]

next_month_names = df.iloc[order_ahead_months]['month'].tolist()
month_weather_pairs = [
    f"{month}: {weather}" for month, weather in zip(next_month_names, next_season_df['weather'])
]
weather_context = ", ".join(month_weather_pairs)

prompt = f"""
The upcoming three months to prepare for are: {weather_context}.
Suggest Amazon packaging materials that would be suitable for storing, shipping, or protecting products in this seasonal context.
Respond in JSON format with item names, categories, and usage recommendations. Give ONLY JSON, AND ALWAYS JSON.
"""

llm = Ollama(model="llama3:latest")
response = llm.invoke(prompt)
parsed_data = parse_json_markdown(response)

items = [entry.get("item", "Unnamed Item") for entry in parsed_data]
item_list_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

# Step 6: Compose email
EMAIL_ADDRESS = "f20230254@dubai.bits-pilani.ac.in"
EMAIL_PASSWORD = "zxje eoqp rnxa vrxu"  # Note: for security, use environment variables in production

msg = EmailMessage()
msg['Subject'] = 'Quarterly Packaging Material Forecast'
msg['From'] = EMAIL_ADDRESS
msg['To'] = 'kodithyalasaiuday1234@gmail.com'
msg.set_content(f"""Hello,

Based on the upcoming seasonal context ({weather_context}), here are the recommended packaging materials to pre-order:

{item_list_text}

Best regards,
Your AI Logistics Assistant
""")

# Step 7: Send email
with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    smtp.send_message(msg)

print("✅ Email sent with packaging item list.")
