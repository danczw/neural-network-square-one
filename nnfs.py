inputs = [1, 2, 3, 2.5]

weightsOne = [0.2, 0.8, -0.5, 1]
weightsTwo = [0.5, -0.91, 0.26, -0.5]
weightsThree = [-0.26, -0.27, 0.17, 0.87]

biasOne = 2
biasTwo = 3
biasThree = 0.5

output = [inputs[0] * weightsOne[0] + inputs[1] * weightsOne[1] + inputs[2] * weightsOne[2] + inputs[3] * weightsOne[3] + biasOne,
         inputs[0] * weightsTwo[0] + inputs[1] * weightsTwo[1] + inputs[2] * weightsTwo[2] + inputs[3] * weightsTwo[3] + biasTwo,
         inputs[0] * weightsThree[0] + inputs[1] * weightsThree[1] + inputs[2] * weightsThree[2] + inputs[3] * weightsThree[3] + biasThree]

print(output)