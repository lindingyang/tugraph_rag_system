from openai import OpenAI


def get_gpt_response(prompt):
    client = OpenAI(
        base_url="your_url",
        api_key="your_api_key"
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="gpt-4o-mini",
            top_p=0,
            temperature=0,
            max_tokens=8192,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return str(e)
