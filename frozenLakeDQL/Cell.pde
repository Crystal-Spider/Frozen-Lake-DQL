class Cell
{
  int x; //Cell x coordinate.
  int y; //Cell y coordinate.
  color c; //Cell color.
  int kind; 
  //Cell kinds:
  /*
    1: Start
    0: Hole
    8: Ice
    3: Goal
  */
  
  public Cell(int x, int y, color c, int kind)
  {
    this.x = x;
    this.y = y;
    this.c = c;
    this.kind = kind;
  }
  
  //Draw cell.
  public void render()
  {
    fill(c);
    rect(x, y, cellDim, cellDim);
  }
}
