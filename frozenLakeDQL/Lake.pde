class Lake
{
  ArrayList<Cell> cells = new ArrayList<Cell>(); //ArrayList to store game cells.
  
  color ice = color(230,248,255); //Ice cells color.
  color hole = color(25,25,112); //Hole cells color.
  
  //Set color, coordinates and kind of each cell.
  public Lake()
  {
    //Set all cells to Ice.
    for(int r = 0; r < sideLength; r++)
    {
      for(int c = 0; c < sideLength; c++)
      {
        cells.add(new Cell(cellDim*c, cellDim*r, ice, 8));
      }
    }
    
    //Set the first element of the ArrayList as the Start cell.
    cells.get(0).c = color(150,0,0);
    cells.get(0).kind = 1;
    
    //Set the first element of the ArrayList as the Goal cell.
    cells.get(cellNum-1).c = color(0,120,0);
    cells.get(cellNum-1).kind = 3;
    
    /*
      Set randomly some cells as holes.
      The cell to the right of the Start cell and the cell above the Goal cell are avoided and cannot become holes.
      This generation is faulty: some holes may be overlapped and it's still possible that the holes are generated in such way
      that they block any possible path from Start to Goal cells. The bigger the map the better this generation works,
      the smaller the map the worse this generation works.
    */
    ArrayList<Integer> holesPos = new ArrayList<Integer>();
    for(int c = 0; c < sideLength; c++)
    {
      holesPos.add((int)random(2, cellNum-2));
    }
    for(int c = 2; c < cellNum-2; c++)
    {
      if(c != sideLength && c != cellNum-sideLength-1 && holesPos.contains(c))
      {
        cells.get(c).c = hole;
        cells.get(c).kind = 0;
      }
    }
  }
  
  //Returns the reward for the action taken by the Agent.
  /*
    If either the Agent took a forbidden action (went out of the map),
    lost the game (went on a hole) or won the game (went on Goal),
    set the attribute agent.done = true (game completed).
  */
  //Updates gameData.
  public double moved()
  {
    if(agent.x < 0 || agent.y < 0 || agent.x > size || agent.y > size) //Out of map.
    {
      gameData.increaseLosses();
      agent.done = true;
      agent.outOfMap = true;
      return -10;
    }
    else if((agent.x - cellDim/2) == cells.get(cells.size()-1).x && (agent.y - cellDim/2) == cells.get(cells.size()-1).y) //Goal.
    {
      gameData.increaseWins();
      if(gameData.gameMoves < gameData.bestMoves)
      {
        gameData.bestMoves = gameData.gameMoves;
      }
      agent.done = true;
      return 10;
    }
    else
    {
      boolean inHole = false;
      for(Cell cell : cells)
      {
        if((agent.x - cellDim/2) == cell.x && (agent.y - cellDim/2) == cell.y && cell.kind == 0) //Hole.
        {
          inHole = true;
        }
      }
      if(inHole)
      {
        gameData.increaseLosses();
        agent.done = true;
        return -10;
      }
      else //Ice.
      {
        return -0.1;
      }
    }
  }
  
  //Draw the Environment.
  public void render()
  {
    for(Cell cell : cells)
    {
      cell.render();
    }
  }
}
