import openai


async def gpt_filter(word_list, prompt_num, model="gpt-4-1106-preview"):
    try:
        completion = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {
                    "role": "user", "content":
                    "This is a list of OCR results:\n"
                    f"{word_list}\n"
                    "Now, I want to use these results as prompts to help Whisper to do speech recognition."
                    "Please filter the list to make it more streamlined by following criteria.\n"
                    "- Only keep easily misspelled words (name of individual, name of organization, "
                    "technical terminology).\n"
                    "- Don't keep wrong English words since the OCR results contain quite a few errors."
                    "For example, toonnetntmenmannonnb,  furatioralty, and fhehedson are not correct English words, "
                    "and they should not be included in the list.\n"
                    f"- Only return up to {prompt_num} words.\n"
                    "- Only return the word list separated by ', ', don't say anything else.\n"
                    "- All words you return should be lowercase.\n"
                    "- If the list of OCR results is empty ('<EMPTY>'), just return NONE."
                }
            ]
        )
        response = completion.choices[0].message["content"]
        if response == "NONE":
            return None
        else:
            return response
    except Exception:
        return gpt_filter(word_list, prompt_num, model)
