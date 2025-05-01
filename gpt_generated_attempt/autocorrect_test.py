from textblob import TextBlob

Corrected = str(TextBlob("I HAVV A DREEM").correct())
print(Corrected)  # Output: I have a dream