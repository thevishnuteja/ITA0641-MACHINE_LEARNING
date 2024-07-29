def find_s(training_data):
    hypothesis = [None] * len(training_data[0][0])
    for example in training_data:
        attributes, label = example
        if label == 'Yes':
            if hypothesis == [None] * len(attributes):
                hypothesis = list(attributes)  
            else:
                for i in range(len(attributes)):
                    if hypothesis[i] != attributes[i]:
                        hypothesis[i] = '?'
    
    return hypothesis
training_data = [
    (('Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'), 'Yes'),
    (('Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'), 'Yes'),
    (('Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'), 'No'),
    (('Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'), 'Yes')
]

# Find the most specific hypothesis
hypothesis = find_s(training_data)
print("The most specific hypothesis is:", hypothesis)
