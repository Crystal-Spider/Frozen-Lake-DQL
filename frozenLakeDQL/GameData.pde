class GameData
{
  int gamesThreshold; //Numbers of games to play before considering the Agent trained.
  
  int trainingGames; //Number of games played while training.
  int wonTrainingGames; //Number of games won while training.
  int lostTrainingGames; //Number of games lost while training.
  
  int playingGames; //Number of games played while playing.
  int wonPlayingGames; //Number of games won while playing.
  int lostPlayingGames; //Number of games lost while playing.
  
  short gameMoves; //Number of moves done in the current game.
  short bestMoves; //Lowest number of moves done in a winning game.
  
  public GameData(int gamesThreshold)
  {
    this.gamesThreshold = gamesThreshold;
    
    trainingGames = 0;
    wonTrainingGames = 0;
    lostTrainingGames = 0;
    
    playingGames = 0;
    wonPlayingGames = 0;
    lostPlayingGames = 0;
    
    gameMoves = 0;
    bestMoves = Short.MAX_VALUE; //Set to the highest value so that this attribute is surely set to the number of moves done in the first game the Agent wins.
  }
  
  public void printData()
  {
    printTrainingData();
    printPlayingData();
    printMovesData();
  }
  
  public void printTrainingData()
  {
    println("Played games (training): " + trainingGames);
    println("Won games (training): " + wonTrainingGames);
    println("Lost games (training): " + lostTrainingGames);
    println();
  }
  
  public void printPlayingData()
  {
    println("Played games (playing): " + playingGames);
    println("Won games (playing): " + wonPlayingGames);
    println("Lost games (playing): " + lostPlayingGames);
    println();
  }
  
  public void printMovesData()
  {
    float d = frame/div;
    if(gameData.trainingGames < gameData.gamesThreshold)
    {
      d = 1;
    }
    println("Moves per second: " + (frameRate/d));
    println("Moves done: " + gameMoves);
    println("Best moves: " + bestMoves);
    println("-----------------------------------------");
  }
  
  public void increaseGames()
  {
    if(gameData.trainingGames < gameData.gamesThreshold)
    {
      trainingGames++;
    }
    else
    {
      playingGames++;
    }
  }
  
  public void increaseWins()
  {
    if(gameData.trainingGames < gameData.gamesThreshold)
    {
      wonTrainingGames++;
    }
    else
    {
      wonPlayingGames++;
    }
  }
  
  public void increaseLosses()
  {
    if(gameData.trainingGames < gameData.gamesThreshold)
    {
      lostTrainingGames++;
    }
    else
    {
      lostPlayingGames++;
    }
  }
}
