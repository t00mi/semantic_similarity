# Importing spacy module and copying all the code extracts from Task 38 pdf.

import spacy

nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
word4 = nlp("dog")
word5 = nlp("milk")
word6 = nlp("bone")
word7 = nlp("cow")

print("cat and monkey")
print(word1.similarity(word2))
print("monkey and banana")
print(word2.similarity(word3))
print("banana and cat")
print(word3.similarity(word1))
print("cat and dog")
print(word1.similarity(word4))
print("monkey and dog")
print(word2.similarity(word4))
print("banana and dog")
print(word3.similarity(word4))
print("cat and milk")
print(word1.similarity(word5))
print("dog and bone")
print(word4.similarity(word6))
print("dog and cow")
print(word4.similarity(word7))
print("monkey and milk")
print(word2.similarity(word5))
print("milk and cow")
print(word5.similarity(word7))

# A note about similarities between cat, monkey, banana and some other examples:

# Generally animals are more similar to each other than when compared to similarity of animals to unanimated objects (banana, milk or bone).
# Interestingly, when it comes to animals, when a dog got added to the equation, he was most similar by far with a cat and least similar with a monkey.
# Actually, this was the highest similarity among compared objects. The reason might be that they are both four legged, domesticated animals.
# Originally, the biggest similarity between an animal and an object was between a monkey and a banana.
# Bananas are the part of these animals diet, so that might be the reason behind this similarity.
# This has changed dramatically when a cow and milk were added to the compared objects. They were even more similar than a monkey and a banana.
# The calves drink milk but cows drink water, however, cows also produce/give milk and they are highly associated with this product.
# This might be the logic behind such high similarity. 
# I did decide to compare some other animals with objects they are sometimes associated with. Suprisingly a dog with a bone and a cat with milk
# returned very low similarity. 
# In overall, the program seems to catch similarites between compared objects in rather accurate way. Having said that, some pairs of objects 
# which function in general view as maybe not so much similar, but linked, they returned really low score on the scale of similarity.

tokens = nlp('cat apple monkey banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I've lost my car in my car",
"I'd like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# A note about differences between the simpler language model ‘en_core_web_sd’ and the model 'en_core_web_md'.

# Straight after running the simpler model 'en_core_web_sm' on the above examples in a different file, the program informed about 
# not using word vectors which were not loaded. Instead, the results were based on context-sensitive tensors.
# When compared returned outputs from the two modules, following observations were made:
# The simpler module returned compared sentences as even less similar than the more advanced one. This might be because
# the more advanced module is able to make abstract connections between words/sentences, as opposed to only direct ones from 
# the context made by the inferior module. However, even greater difference could be observed when looking at similarity results between words/objects.
# The simpler module did return pretty much the same similarity between animals as the more complex one. Interesting is the return on the similarity
# between fruits (apple/banana) which dropped significantly. After running the simpler module, the similarity between words monkey and banana dropped 
# as well. Most interestingly, similarity of a word apple with a cat and a monkey did rise by a lot. When comparing all the results, one might have 
# the impression that the more advanced module is more accurate when it comes to assessing similarity between words. Even though, the simpler module 
# returned comparable similarites for sentences and for words describing animals, other comparisons are very tough to explain and they seemed not consistent.      