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
    neurons[1] = neurons[0].dotProduct(weightsOutputNeurons).addition(biasOutputNeurons).sigmoid();
    
    return neurons;
  }
  
  public void learn(Layer inputNeurons, Layer[] neurons, Layer desiredOutput)
  {
    Layer hiddenNeurons = neurons[0];
    Layer outputNeurons = neurons[1];
    
    /*
    First letter:
      I = Input to neurons (value passed times the weights plus the bias)
      W = Weights
      O = Output of neurons (Input with activation function)
      C = Cost (Loss)
      
    Second letter:
      I = Input Neurons
      H = Hidden Neurons
      L = Last Neurons (Output Neurons)
      
    Miscellaneous:
      dy/dx = Notation for derivative
    */
    
    /*
    OI = inputNeurons
    
    WH = weightsHiddenNeurons
    IH = inputNeurons.layerMultiplication(weightsHiddenNeurons).addition(biasHiddenNeurons) (OI*WH+BH)
    OH = hiddenNeurons (IH.sigmoid())
    
    WL = weightsOutputNeurons
    IL = hiddenNeurons.layerMultiplication(weightsOutputNeurons).addition(biasOutputNeurons) (OH*WL+BO)
    OL = outputNeurons (IL.sigmoid())
    
    C  = Math.pow(outputNeurons.subtraction(desiredOutput), 2) (OL-DO)^2
    */
    
    /*
    Calculations for dC/dWL:
      dIL/dWL = (OH*WL+BO)' = (OH*WL)' + 0 = OH
      dOL/dIL = (OL)' = OL.sigmoidDerivative() |OR| (IL.sigmoid())' = IL.sigmoidDerivative()
      dC/dOL  = ((OL-DO)^2)' = 2(OL-DO)*1 = 2(OL-DO)
    
    Pseudocode:
      dIL/dWL = hiddenNeurons;
      dOL/dIL = outputNeurons.sigmoidDerivative();
      dC/dOL  = outputNeurons.subtraction(desiredOutput).valueMultiplication(2);
    */
    Layer lossDerivative = outputNeurons.subtraction(desiredOutput).valueMultiplication(2);
    Layer partialChange = outputNeurons.sigmoidDerivative().layerMultiplication(lossDerivative);
    Layer totalChangeWeightsOutputNeurons = new Layer(weightsOutputNeurons.hei, weightsOutputNeurons.wid);
    for(int h = 0; h < totalChangeWeightsOutputNeurons.hei; h++)
    {
      for(int o = 0; o < totalChangeWeightsOutputNeurons.wid; o++)
      {
        totalChangeWeightsOutputNeurons.layer[h][o] = hiddenNeurons.layer[0][h] * partialChange.layer[0][o];
      }
    }
    weightsOutputNeurons = weightsOutputNeurons.subtraction(totalChangeWeightsOutputNeurons.valueMultiplication(learningRate));
    
    /*
    Calculations for dC/dBO:
      dIL/dBO = (OH*WL+BO)' = 0 + (BO)' = 1
      dOL/dIL = (OL)' = OL.sigmoidDerivative() |OR| (IL.sigmoid())' = IL.sigmoidDerivative()
      dC/dOL  = ((OL-DO)^2)' = 2(OL-DO)*1 = 2(OL-DO)
    
    Pseudocode:
      dIL/dBO = 1; //No actual code needed
      dOL/dIL = outputNeurons.sigmoidDerivative();
      dC/dOL  = outputNeurons.subtraction(desiredOutput).valueMultiplication(2);
    */
    biasOutputNeurons = biasOutputNeurons.subtraction(partialChange);
    
    /*
    Calculations for dC/dWH:
      dIH/dWH = (OI*WH+BH)' = OI
      dOH/dIH = (OH)' = OH.sigmoidDerivative() |OR| (IH.sigmoid())' = IH.sigmoidDerivative()
      dC/dOH  = dC(OL1)/dOH + dC(OL2)/dOH + ... + dC(OLn)/dOH
      dC(OL)/dOH = dIL/dOH * dC(OL)/dIL
      
      dIL/dOH = (OH*WL+BO)' = WL
      dC(OL)/dIL = dOL/dIL * dC/dOL
      
      dOL/dIL = (OL)' = OL.sigmoidDerivative() |OR| (IL.sigmoid())' = IL.sigmoidDerivative()
      dC/dOL  = ((OL-DO)^2)' = 2(OL-DO)*1 = 2(OL-DO)
    
    Pseudocode:
      dIH/dWH = inputNeurons;
      dOH/dIH = hiddenNeurons.sigmoidDerivative();
      dC/dOH  = changeSummation;
      dC(OL)/dOH = s;
      
      dIL/dOH = weightsOutputNeurons;
      dC(OL)/dIL = outputNeurons.sigmoidDerivative().layerMultiplication(lossDerivative); //partialChange
      
      dOL/dIL = outputNeurons.sigmoidDerivative();
      dC/dOL  = outputNeurons.subtraction(desiredOutput).valueMultiplication(2);
    */
    Layer changeSummation = new Layer(weightsOutputNeurons.hei, 1);
    for(int h = 0; h < changeSummation.hei; h++)
    {
      double s = 0;
      for(int w = 0; w < changeSummation.wid; w++)
      {
        s += weightsOutputNeurons.layer[h][w] * partialChange.layer[0][w];
      }
      changeSummation.layer[h][0] = s;
    }
    Layer totalChangeWeightsHiddenNeurons = new Layer(weightsHiddenNeurons.hei, weightsHiddenNeurons.wid);
    for(int h = 0; h < totalChangeWeightsHiddenNeurons.hei; h++)
    {
      for(int w = 0; w < totalChangeWeightsHiddenNeurons.wid; w++)
      {
        totalChangeWeightsHiddenNeurons.layer[h][w] = inputNeurons.layer[0][h] * hiddenNeurons.sigmoidDerivative().layer[0][w] * changeSummation.layer[w][0];
      }
    }
    weightsHiddenNeurons = weightsHiddenNeurons.subtraction(totalChangeWeightsHiddenNeurons.valueMultiplication(learningRate));
    
    /*
    Calculations for dC/dBH:
      dIH/dBH = (OI*WH+BH)' = 1
      dOH/dIH = (OH)' = OH.sigmoidDerivative() |OR| (IH.sigmoid())' = IH.sigmoidDerivative()
      dC/dOH  = dC(OL1)/dOH + dC(OL2)/dOH + ... + dC(OLn)/dOH
      dC(OL)/dOH = dIL/dOH * dC(OL)/dIL
      
      dIL/dOH = (OH*WL+BO)' = WL
      dC(OL)/dIL = dOL/dIL * dC/dOL
      
      dOL/dIL = (OL)' = OL.sigmoidDerivative() |OR| (IL.sigmoid())' = IL.sigmoidDerivative()
      dC/dOL  = ((OL-DO)^2)' = 2(OL-DO)*1 = 2(OL-DO)
    
    Pseudocode:
      dIH/dBH = 1;
      dOH/dIH = hiddenNeurons.sigmoidDerivative();
      dC/dOH  = changeSummation;
      dC(OL)/dOH = s;
      
      dIL/dOH = weightsOutputNeurons;
      dC(OL)/dIL = outputNeurons.sigmoidDerivative().layerMultiplication(lossDerivative); //partialChange
      
      dOL/dIL = outputNeurons.sigmoidDerivative();
      dC/dOL  = outputNeurons.subtraction(desiredOutput).valueMultiplication(2);
    */
    Layer totalChangeBiasHiddenNeurons = new Layer(biasHiddenNeurons.hei, biasHiddenNeurons.wid);
    for(int h = 0; h < totalChangeBiasHiddenNeurons.hei; h++)
    {
      for(int w = 0; w < totalChangeBiasHiddenNeurons.wid; w++)
      {
        totalChangeBiasHiddenNeurons.layer[h][w] = hiddenNeurons.sigmoidDerivative().layer[0][w] * changeSummation.layer[w][0];
      }
    }
    biasHiddenNeurons.subtraction(totalChangeBiasHiddenNeurons);
    
    
    
    
    
    /*
    Layer dBiasO = outputNeurons.subtraction(desiredOutput).layerMultiplication(hiddenNeurons.dotProduct(weightsOutputNeurons).addition(biasOutputNeurons).sigmoidDerivative());
    
    Layer dBiasH = dBiasO.dotProduct(weightsOutputNeurons.transpose()).layerMultiplication(inputNeurons.dotProduct(weightsHiddenNeurons).addition(biasHiddenNeurons).sigmoidDerivative());
    
    Layer dWeightH = inputNeurons.transpose().dotProduct(dBiasH);
    Layer dWeightO = hiddenNeurons.transpose().dotProduct(dBiasO);
    
    //Set new values for Weights and Biases
    weightsHiddenNeurons = weightsHiddenNeurons.subtraction(dWeightH.valueMultiplication(learningRate));
    biasHiddenNeurons = biasHiddenNeurons.subtraction(dBiasH.valueMultiplication(learningRate));
    weightsOutputNeurons = weightsOutputNeurons.subtraction(dWeightO.valueMultiplication(learningRate));
    biasOutputNeurons = biasOutputNeurons.subtraction(dBiasO.valueMultiplication(learningRate));
    */
  }
}
