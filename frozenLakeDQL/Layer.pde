class Layer
{
  int hei;
  int wid;
  double[][] layer;
  
  public Layer(int hei, int wid)
  {
    this.hei = hei;
    this.wid = wid;
    layer = new double[hei][wid];
    
    for(int h = 0; h < hei; h++)
    {
      for(int w = 0; w < wid; w++)
      {
        layer[h][w] = 0;
      }
    }
  }
  
  Layer(double[][] m)
  {
    this.hei = m.length;
    this.wid = m[0].length;
    this.layer = m;
  }

  Layer(double[] m)
  {
    this.hei = 1;
    this.wid = m.length;
    this.layer = new double[this.hei][m.length];

    for(int c = 0; c < m.length; c++)
    {
        this.layer[0][c] = m[c];
    }
  }
  
  public void randomize()
  {
    for(int h = 0; h < hei; h++)
    {
      for(int w = 0; w < wid; w++)
      {
        layer[h][w] = Math.random() * 0.001;
      }
    }
  }
  
  public Layer sigmoid()
  {
    Layer result = new Layer(this.hei, this.wid);
    
    for(int h = 0; h < hei; h++)
    {
      for(int w = 0; w < wid; w++)
      {
        result.layer[h][w] = 1 / (1 + Math.exp(-this.layer[h][w]));
      }
    }
    
    return result;
  }
  
  public Layer sigmoidDerivative()
  {
    Layer result = new Layer(this.hei, this.wid);
    
    for(int h = 0; h < hei; h++)
    {
      for(int w = 0; w < wid; w++)
      {
        result.layer[h][w] = Math.exp(-this.layer[h][w]) / (Math.pow(1 + Math.exp(-this.layer[h][w]), 2));
      }
    }
    
    return result;
  }

  public Layer dotProduct(Layer m)
  {
    Layer result = new Layer(this.hei, m.wid);

    if(this.wid == m.hei)
    {
      double s = 0;

      for(int h = 0; h < this.hei; h++)
      {
        for(int mw = 0; mw < m.wid; mw++)
        {
          for(int w = 0; w < this.wid; w++)
          {
            s += this.layer[h][w] * m.layer[w][mw];
          }
          result.layer[h][mw] = s;
          s = 0;
        }
      }
    }
    else
    {
      System.out.println("DotProduct impossibile.");
    }
    
    return result;
  }
  
  public Layer addition(Layer m)
  {
    Layer result = new Layer(this.hei, this.wid);

    if(m.hei == this.hei && m.wid == this.wid)
    {
      for(int h = 0; h < this.hei; h++)
      {
        for(int w = 0; w < this.wid; w++)
        {
          result.layer[h][w] = this.layer[h][w] + m.layer[h][w];
        }
      }
    }
    else
    {
      System.out.println("Addition impossible.");
    }
    
    return result;
  }
  
  public Layer subtraction(Layer m)
  {
    Layer result = new Layer(this.hei, this.wid);

    if(m.hei == this.hei && m.wid == this.wid)
    {
      for(int h = 0; h < this.hei; h++)
      {
        for(int w = 0; w < this.wid; w++)
        {
          result.layer[h][w] = this.layer[h][w] - m.layer[h][w];
        }
      }
    }
    else
    {
      System.out.println("Subtraction impossible.");
    }
    
    return result;
  }
  
  public Layer valueMultiplication(double value)
  {
    Layer result = new Layer(this.hei, this.wid);

    for (int h = 0; h < this.hei; h++)
    {
      for (int w = 0; w < this.wid; w++)
      {
        result.layer[h][w] = this.layer[h][w] * value;
      }
    }

    return result;
  }
  
  public Layer layerMultiplication(Layer m)
  {
    Layer result = new Layer(this.hei, this.wid);

    if(this.hei == m.hei && this.wid == m.wid)
    {
      for(int h = 0; h < this.hei; h++)
      {
        for (int w = 0; w < this.wid; w++)
        {
          result.layer[h][w] = this.layer[h][w] * m.layer[h][w];
        }
      }
    }
    else
    {
      System.out.println("Moltiplicazione impossibile, le matrici non hanno le stesse dimensioni");
    }
    return result;
  }
  
  public Layer transpose()
  {
    Layer result = new Layer(this.wid, this.hei);

    for (int w = 0; w < this.wid; w++)
    {
      for (int h = 0; h < this.hei; h++)
      {
        result.layer[w][h] = this.layer[h][w];
      }
    }

    return result;
  }
  
  public int getMaxPosition()
  {
    int action = 0;
    double max = layer[0][0];
    
    for(int c = 1; c < 4; c++)
    {
      if(layer[0][c] > max)
      {
        max = layer[0][c];
        action = c;
      }
      else if(layer[0][c] == max && random(0, 1) > 0.5)
      {
        max = layer[0][c];
        action = c;
      }
    }
    
    return action;
  }
  
  public double getMaxValue()
  {
    return layer[0][getMaxPosition()];
  }
}
