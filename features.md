# Language learning app 

## Flash card feature 
- JSON of vocab has things like Kanji, Kana, the type of word, and the meaning
- Have Boolean to determine whether or not to show kana (kana should be in smaller font above the respective kanji)
- Have a button to show an example sentence, maybe use a TTS model to speak it 
- As you view flash cards, they get added to your word bank, other features utilize that to determine which words to choose from
- All flash cards should have a verbose version with different conjugations and verb-stem to be more easily searchable by other features if used with different grammar

## Word matching feature 
- Like Quizlet: have a grid of words and their meanings, each word is a button and the last two words pressed are “matched”
- Have a kana and non-kana option 
- Words chosen in a weighted random order; as they show up more, their weight decreases and they become less common
- Should be able to say “I want to see this less” or “I want to see this more”

## Vocab speaking practice feature 
- Like Duolingo, but with different formats
- English meaning displayed / spoken -> respond with Japanese word 
- Japanese word displayed / spoken -> respond with English meaning 
  - This will be harder to score since one thing can mean many different things
  - Some sort of concept mapping with similarity search or maybe there are some NLP models/methods that can do this 
- Option to make it written instead of spoken
- Kana option 

## Practical usage feature 
- Given a word, write or speak a sentence using it 
- Kana option 

## Writing practice feature 
- Not quite how Duolingo does it, where it outlines it for you and makes it stay within a certain area
- Use OCR or template matching model to compare to the ground truth
- Track events while writing, each stroke should be in the correct direction and the correct order (basically tracking velocity)
- If the output matches the kanji/kana, then it is passing, but is only given perfect score depending on strokes

## Conversation feature 
- Based on either word bank, or n-level, voice agent will speak to you with limited vocabulary
- Should be able to see the words as they are spoken with links to flash cards in case