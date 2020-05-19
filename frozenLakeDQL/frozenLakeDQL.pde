int size = 900; //Canvas size (pixels).
int nnSpace = size*2/3; //Space to draw the Agent's Neural Network.
int sideLength = 4; //Cells number per row/column (To make the game work, this value must be >= 2).
int cellNum = sideLength*sideLength; //Total cells number.
int cellDim = size/sideLength; //Single cell dimension.
int frame = 1000; //FPS.
float div = 10; //To slow down Agent while playing.

//cellNum - gamesThreshold (experimental values)
/*
        2 - ?
        3 - ?
        4 - 5000
        5 - ?
        6 - ?
        7 - ?
        8 - ?
        9 - ?
       10 - ?
       11 - ?
       12 - ?
       13 - ?
       14 - ?
       15 - ?
       16 - ?
*/

GameData gameData = new GameData(5000); //Game infos.
Lake lake = new Lake(); //Environment.
Agent agent = new Agent(cellDim/2); //Agent.

void settings()
{
  size(size+nnSpace, size);
}

void setup()
{
  frameRate(frame);
  background(0);
  smooth();
}

void draw()
{
  if(gameData.trainingGames < gameData.gamesThreshold)
  {
    agent.train();
  }
  else
  {
    if(frameCount%(frame/div) == 0) //To slow down the agent.
    {
      agent.play();
    }
  }
}
