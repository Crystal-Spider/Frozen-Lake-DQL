class Brain
{
  //Neural Network Variables.
  private Layer weightsHiddenNeurons;
  private Layer biasesHiddenNeurons;
  
  private Layer weightsOutputNeurons;
  private Layer biasesOutputNeurons;
  
  private double learningRate;
  
  //Drawing Variables.
  private float centerInput;
  private float distanceInput;
  
  private float centerHidden;
  private float distanceHidden;
  
  private float centerOutput;
  private float distanceOutput;
  
  color lowColor;
  color highColor;
  
  public Brain(int inputNeuronsNumber, int hiddenNeuronsNumber, int outputNeuronsNumber, double learningRate)
  {
    this.learningRate = learningRate;
    
    weightsHiddenNeurons = new Layer(inputNeuronsNumber, hiddenNeuronsNumber);
    biasesHiddenNeurons = new Layer(1, hiddenNeuronsNumber);
    
    weightsOutputNeurons = new Layer(hiddenNeuronsNumber, outputNeuronsNumber);
    biasesOutputNeurons = new Layer(1, outputNeuronsNumber);
    
    /*weightsHiddenNeurons.randomize();
    biasesHiddenNeurons.randomize();
    weightsOutputNeurons.randomize();
    biasesOutputNeurons.randomize();*/
    
    setDrawingVariables();
  }
  
  public Layer[] estimateOutput(double[] input)
  {
    Layer[] neurons = new Layer[2]; //Hidden neurons [0] and Output neurons [1].
    neurons[0] = new Layer(input); //To be transformed into Hidden neurons.
    
    //Hidden neurons values calculation.
    neurons[0] = neurons[0].dotProduct(weightsHiddenNeurons).addition(biasesHiddenNeurons).sigmoid();
    
    //Output neurons values calculation.
    neurons[1] = neurons[0].dotProduct(weightsOutputNeurons).addition(biasesOutputNeurons);
    
    this.render(new Layer(input), neurons[0], neurons[1].sigmoid()); //Draw Agent's Neural Network.
    
    return neurons;
  }
  
  public void learn(Layer inputNeurons, Layer[] neurons, Layer desiredOutput)
  {
    Layer hiddenNeurons = neurons[0];
    Layer outputNeurons = neurons[1];
    
    /*
    Symbols:
      IN = Input Neurons (Output of Input Neurons) - InputNeurons;
      
      WH = Weights of Hidden Neurons - WeightsHiddenNeurons;
      BH = Bias of Hidden Neurons - BiasHiddenNeurons;
      IH = Input to Hidden Neurons [ IN*WH+BH ] - (No variable associated);
      OH = Output of Hidden Neurons [ Sigmoid(IH) ] - HiddenNeurons;
      
      WO = Weights of Output Neurons - WeightsOutputNeurons;
      BO = Bias of Output Neurons - BiasOutputNeurons;
      IO = Input to Output Neurons [ OH*WO+BO ] - OutputNeurons;
      OO = Output of Output Neurons [ IO ] (No activation function is used);
      
      DO = Desired Output - DesiredOutput;
    
      LS = Loss Function [ (DO-OO)^2 ];
      
      dy/dx = Derivative notation.
      
    Calculations for dLS/dWO:
      dLS/dWO = dLS/dOO * dOO/dIO * dIO/dWO = dLS/dOO * 1 * dIO/dWO = dLS/dOO * dIO/dWO
      
      dLS/dOO = ((DO-OO)^2)' = 2(DO-OO)*(-1) = 2(OO-DO)
      dIO/dWO = (OH*WO+BO)' = OH
      
      dLS/dWO = OH*2(OO-DO)
      
    Pseudocode for dLS/dWO:
      dLS/dOO = (outputNeurons - desiredOutput)*2;
      dIO/dWO = hiddenNeurons;
      
      dLS/dWO = hiddenNeurons*((outputNeurons-desiredOutput)*2);
      
    Calculations for dLS/dBO:
      dLS/dBO = dLS/dOO * dOO/dIO * dIO/dBO = dLS/dOO * dIO/dBO
      
      dLS/dOO = ((DO-OO)^2)' = 2(DO-OO)*(-1) = 2(OO-DO)
      dIO/dWO = (OH*WO+BO)' = 1
      
      dLS/dWO = 2(OO-DO)
      
    Pseudocode for dLS/dBO:
      dLS/dOO = (outputNeurons - desiredOutput)*2;
      
      dLS/dWO = (outputNeurons-desiredOutput)*2;
      
    Calculations for dLS/dWH:
      dLS/dWH = dLS/dOO * dOO/dIO * dIO/dOH * dOH/dIH * dIH/dWH = dLS/dOO * dIO/dOH * dOH/dIH * dIH/dWH
      
      dLS/dOO = 2(OO-DO)
      dIO/dOH = (OH*WO+BO)' = WO
      dOH/dIH = (Sigmoid(IH))' = SigmoidDerivative(IH)*1 = SigmoidDerivative(IH)
      dIH/dWH = (IN*WH+BH)' = IN
      
      dLS/dWH = 2(OO-DO)*WO*SigmoidDerivative(IH)*IN
      
    Pseudocode for dLS/dWH:
      dLS/dOO = (outputNeurons - desiredOutput)*2;
      dIO/dOH = WeightsOutputNeurons;
      dOH/dIH = SigmoidDerivative(InputNeurons*WeightsHiddenNeurons+BiasHiddenNeurons);
      dIH/dWH = InputNeurons;
      
      dLS/dWH = 2(OO-DO)*WO*SigmoidDerivative(IH)*IN
      
    Calculations for dLS/dBH:
      dLS/dBH = dLS/dOO * dOO/dIO * dIO/dOH * dOH/dIH * dIH/dBH = dLS/dOO * dIO/dOH * dOH/dIH * dIH/dBH
      
      dLS/dOO = 2(OO-DO)
      dIO/dOH = (OH*WO+BO)' = WO
      dOH/dIH = (Sigmoid(IH))' = SigmoidDerivative(IH)*1 = SigmoidDerivative(IH)
      dIH/dWH = (IN*WH+BH)' = 1
      
      dLS/dWH = 2(OO-DO)*WO*SigmoidDerivative(IH)
      
    Pseudocode for dLS/dBH:
      dLS/dOO = (outputNeurons - desiredOutput)*2;
      dIO/dOH = WeightsOutputNeurons;
      dOH/dIH = SigmoidDerivative(InputNeurons*WeightsHiddenNeurons+BiasHiddenNeurons);
      
      dLS/dWH = 2(OO-DO)*WO*SigmoidDerivative(IH)
    */
    
    /*
      The Loss Derivative equals dLS/dBO (in this case), so the variable for its value
      is not instantiated and instead the variable dBiasO is used.
      
      The only difference between dLS/dWH (dWeightH) and dLS/dBH (dBiasH) is that the
      latter is not multiplied with IN (InputNeurons). So, to save computational time,
      dWeightH just equals dBiasH multiplied with InputNeurons.
      
      Multiplications within the chain rule are all dot products when matrices sizes
      are different.
      
      In certain cases in the code the transpose function is used on some Layers
      to make the dot product with another Layer possible. Note that transposing
      a matrix doesn't change its values but only swaps rows and columns.
    */
    
    Layer dBiasO = (outputNeurons.subtraction(desiredOutput)).valueMultiplication(2);
    Layer dBiasH = (dBiasO.dotProduct(weightsOutputNeurons.transpose())).layerMultiplication((inputNeurons.dotProduct(weightsHiddenNeurons).addition(biasesHiddenNeurons)).sigmoidDerivative());
    
    Layer dWeightO = (hiddenNeurons.transpose()).dotProduct(dBiasO);
    Layer dWeightH = (inputNeurons.transpose()).dotProduct(dBiasH);
    
    //Set new values for Weights and Biases
    weightsHiddenNeurons = weightsHiddenNeurons.subtraction(dWeightH.valueMultiplication(learningRate));
    biasesHiddenNeurons = biasesHiddenNeurons.subtraction(dBiasH.valueMultiplication(learningRate));
    weightsOutputNeurons = weightsOutputNeurons.subtraction(dWeightO.valueMultiplication(learningRate));
    biasesOutputNeurons = biasesOutputNeurons.subtraction(dBiasO.valueMultiplication(learningRate));
  }
  
  public void setDrawingVariables()
  {
    this.centerInput = size + nnSpace/5;
    this.distanceInput = size/(cellNum*2); //Used also as Neurons size.
    
    this.centerHidden = size + nnSpace/2;
    this.distanceHidden = size/cellNum; //(cellNum/2*2)
    
    this.centerOutput = size + nnSpace*4/5;
    this.distanceOutput = size/8; //4*2
    
    lowColor = color(0, 0, 100);
    highColor = color(0, 127, 255);
  }
  
  //Method to draw the Neural Network (Only works well when cellNum <= 8)
  /*
    Lines are the weights. They range from red to yellow, depending on their value.
    Circles are the neurons. They range from black to white depending on their value.
    Circles' margins are the neurons' biases. They range from blue to light blue depending on their value.
  */
  //Biases for Input Neurons never change and are always set to white.
  //With certain values the Input Neurons and the Hidden Neurons are not vertically centered.
  public void render(Layer InputNeurons, Layer HiddenNeurons, Layer OutputNeurons)
  {
    fill(0);
    noStroke();
    rect(size, 0, nnSpace, size);
    
    Layer norm; //Layer used to normalize data.
    
    norm = weightsHiddenNeurons.sigmoid();
    //Draw Hidden Neurons Weights
    strokeWeight(1);
    for(int h = 0; h < norm.hei; h++)
    {
      for(int w = 0; w < norm.wid; w++)
      {
        stroke(gradient(lowColor, highColor, norm.layer[h][w]));
        line(centerInput, distanceInput*h*2 + distanceInput, centerHidden, distanceHidden*w*2 + distanceHidden);
      }
    }
    
    //Draw Input Neurons.
    strokeWeight(2);
    for(int w = 0; w < InputNeurons.wid; w++)
    {
      stroke(highColor);
      fill(gradient(0, 255, InputNeurons.layer[0][w]));
      ellipse(centerInput, distanceInput*w*2 + distanceInput, distanceInput, distanceInput);
    }
    
    norm = weightsOutputNeurons.sigmoid();
    //Draw Output Neurons Weights
    strokeWeight(1);
    for(int h = 0; h < norm.hei; h++)
    {
      for(int w = 0; w < norm.wid; w++)
      {
        stroke(gradient(lowColor, highColor, norm.layer[h][w]));
        line(centerHidden, distanceHidden*h*2 + distanceHidden, centerOutput, distanceOutput*w*2 + distanceOutput);
      }
    }
    
    norm = biasesHiddenNeurons.sigmoid();
    //Draw Hidden Neurons.
    strokeWeight(3);
    for(int w = 0; w < HiddenNeurons.wid; w++)
    {
      stroke(gradient(lowColor, highColor, norm.layer[0][w]));
      fill(gradient(0, 255, HiddenNeurons.layer[0][w]));
      ellipse(centerHidden, distanceHidden*w*2 + distanceHidden, distanceInput, distanceInput);
    }
    
    norm = biasesOutputNeurons.sigmoid();
    //Draw OutputNeurons;
    for(int w = 0; w < OutputNeurons.wid; w++)
    {
      stroke(gradient(lowColor, highColor, norm.layer[0][w]));
      fill(gradient(0, 255, OutputNeurons.layer[0][w]));
      ellipse(centerOutput, distanceOutput*w*2 + distanceOutput, distanceInput, distanceInput);
    }
  }
  
  private color gradient(color c1, color c2, double value)
  {
    return lerpColor(c1, c2, (float)value);
  }
}
