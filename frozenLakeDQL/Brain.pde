class Brain
{
  private Layer weightsHiddenNeurons;
  private Layer biasHiddenNeurons;
  
  private Layer weightsOutputNeurons;
  private Layer biasOutputNeurons;
  
  private double learningRate;
  
  public Brain(int inputNeuronsNumber, int hiddenNeuronsNumber, int outputNeuronsNumber, double learningRate)
  {
    this.learningRate = learningRate;
    
    weightsHiddenNeurons = new Layer(inputNeuronsNumber, hiddenNeuronsNumber);
    biasHiddenNeurons = new Layer(1, hiddenNeuronsNumber);
    
    weightsOutputNeurons = new Layer(hiddenNeuronsNumber, outputNeuronsNumber);
    biasOutputNeurons = new Layer(1, outputNeuronsNumber);
    
    weightsHiddenNeurons.randomize();
    biasHiddenNeurons.randomize();
    weightsOutputNeurons.randomize();
    biasOutputNeurons.randomize();
  }
  
  public Layer[] estimateOutput(double[] input)
  {
    Layer[] neurons = new Layer[2]; //Hidden neurons [0] and Output neurons [1].
    neurons[0] = new Layer(input); //To be transformed into Hidden neurons.
    
    //Hidden neurons values calculation.
    neurons[0] = neurons[0].dotProduct(weightsHiddenNeurons).addition(biasHiddenNeurons).sigmoid();
    
    //Output neurons values calculation.
    neurons[1] = neurons[0].dotProduct(weightsOutputNeurons).addition(biasOutputNeurons);
    
    return neurons;
  }
  
  public void learn(Layer inputNeurons, Layer[] neurons, Layer desiredOutput)
  {
    Layer hiddenNeurons = neurons[0];
    Layer outputNeurons = neurons[1];
    
    Layer lossDerivative = ((outputNeurons.subtraction(desiredOutput)).valueMultiplication(2));
    
    Layer dWeightO = (hiddenNeurons.transpose()).dotProduct(lossDerivative);
    Layer dWeightH = (inputNeurons.transpose()).dotProduct((lossDerivative.dotProduct(weightsOutputNeurons.transpose())).layerMultiplication(((inputNeurons.dotProduct(weightsHiddenNeurons).addition(biasHiddenNeurons)).sigmoidDerivative())));
    
    Layer dBiasH = (lossDerivative.dotProduct(weightsOutputNeurons.transpose())).layerMultiplication(((inputNeurons.dotProduct(weightsHiddenNeurons).addition(biasHiddenNeurons)).sigmoidDerivative()));
    
    //Set new values for Weights and Biases
    weightsHiddenNeurons = weightsHiddenNeurons.subtraction(dWeightH.valueMultiplication(learningRate));
    biasHiddenNeurons = biasHiddenNeurons.subtraction(dBiasH.valueMultiplication(learningRate));
    weightsOutputNeurons = weightsOutputNeurons.subtraction(dWeightO.valueMultiplication(learningRate));
    biasOutputNeurons = biasOutputNeurons.subtraction(lossDerivative.valueMultiplication(learningRate));
    
    println(lossFunction(outputNeurons, desiredOutput));
  }
  
  public double lossFunction(Layer outputNeurons, Layer desiredOutput)
  {
    Layer l = outputNeurons.subtraction(desiredOutput);
    l = l.layerMultiplication(l);
    return l.sumAll();
  }
}
