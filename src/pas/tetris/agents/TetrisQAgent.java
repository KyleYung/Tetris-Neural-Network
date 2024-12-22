package src.pas.tetris.agents;


// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.ArrayList;


// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;


public class TetrisQAgent
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.99;
    private Random random;
    private int linesCleared;
    private int holes;
    private int bumps;
    private List<Integer> heights;
    private int maxHeight;
    private double unevenness;
    private double flatness;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction()
    {
        // System.out.println("initQFunction called!");
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector

        final int inputDim = Board.NUM_ROWS * Board.NUM_COLS + 6;
        final int hidden1 = 128;
        final int hidden2 = 64;
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(inputDim, hidden1));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hidden1, hidden2));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hidden2, outDim));

        return qFunction;
    }

    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */
    @Override
    public Matrix getQFunctionInput(final GameView game,
                                    final Mino potentialAction)
    {
        Matrix boardMatrix = null;
        try {
            boardMatrix = game.getGrayscaleImage(potentialAction);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }

        // Update global variables
        linesCleared = calculateLinesCleared(boardMatrix);
        holes = calculateHoles(boardMatrix);
        bumps = calculateBumps(boardMatrix);
        maxHeight = calculateMaxHeight(boardMatrix);
        heights = calculateColumnHeights(boardMatrix);
        flatness = calculateFlatness(heights);
        unevenness = calculateUnevenness(boardMatrix);

        // Flatten the boardMatrix and append features
        Matrix inputMatrix = Matrix.full(1, Board.NUM_ROWS * Board.NUM_COLS + 6, 0);
        int index = 0;
        inputMatrix.set(0, index++, linesCleared);
        inputMatrix.set(0, index++, holes);
        inputMatrix.set(0, index++, bumps);
        inputMatrix.set(0, index++, maxHeight);
        inputMatrix.set(0, index++, flatness);
        inputMatrix.set(0, index++, unevenness);
        return inputMatrix;
    }

    private int calculateLinesCleared(Matrix boardMatrix) {
        int clearedLines = 0;
        for (int row = 0; row < Board.NUM_ROWS; row++) {
            boolean isComplete = true;
            for (int col = 0; col < Board.NUM_COLS; col++) {
                if (boardMatrix.get(row, col) == 0.0) {
                    isComplete = false;
                    break;
                }
            }
            if (isComplete) {
                clearedLines++;
            }
        }
        return clearedLines;
    }

    private int calculateHoles(Matrix boardMatrix) {
        int holes = 0;
        for (int col = 0; col < Board.NUM_COLS; col++) {
            boolean foundBlock = false;
            for (int row = 0; row < Board.NUM_ROWS; row++) {
                if (boardMatrix.get(row, col) != 0.0) {
                    foundBlock = true;
                } else if (foundBlock) {
                    holes++;
                }
            }
        }
        return holes;
    }

    private int calculateBumps(Matrix boardMatrix) {
        List<Integer> heights = calculateColumnHeights(boardMatrix);
        int bumps = 0;
        for (int i = 1; i < heights.size(); i++) {
            bumps += Math.abs(heights.get(i) - heights.get(i - 1));
        }
        return bumps;
    }

     private int calculateMaxHeight(Matrix board) {
        int maxh = 0;
        for (int col = 0; col < Board.NUM_COLS; col++) {
            int curr = 0;
            for (int row = 0; row < Board.NUM_ROWS; row++) {
                double block = board.get(row, col);
                if (block != 0.0) { 
                    curr = Board.NUM_ROWS - row;
                    if (curr > maxh) {
                        maxh = curr;
                    }
                }
            }
        }
        return maxh;
    }

    //  private  double calculateAvgHeight(Matrix board) {
    //     int total = 0;
    //     for (int col = 0; col < Board.NUM_COLS; col++) {
    //         for (int row = 0; row < Board.NUM_ROWS; row++) {
    //             double block = board.get(row, col);
    //             if (block != 0.0) { 
    //                 total += (Board.NUM_ROWS - row);
    //                 break;
    //             }
    //         }
    //     }
    //     double avg = (double) (total / Board.NUM_COLS);
    //     return avg;
    // }

    private List<Integer> calculateColumnHeights(Matrix boardMatrix) {
        List<Integer> heights = new ArrayList<>();
        for (int col = 0; col < Board.NUM_COLS; col++) {
            int curr = 0;
            for (int row = 0; row < Board.NUM_ROWS; row++) {
                if (boardMatrix.get(row, col) != 0.0) {
                    curr = Board.NUM_ROWS - row;
                    break;
                }
            }
            heights.add(curr);
        }
        return heights;
    }

    private double calculateUnevenness(Matrix boardMatrix) {
        double unevenness = calculateColumnHeights(boardMatrix).stream().mapToDouble(x -> Math.abs(x - 4)).sum();
        return unevenness;
    }

    private double calculateFlatness(List<Integer> c) {
        double averageHeight = c.stream().mapToInt(Integer::intValue).average().orElse(0.0);
        double flatness = 0.0;
        for (int height : c) {
            flatness += Math.abs(height - averageHeight);
        }
        return flatness;
    }

    private int colHeight(Matrix image) {
        int height = 0;
        for (int i = 0; i < Board.NUM_ROWS; i++) {
            for (int j = 0; j < Board.NUM_COLS; j++) {
                if (image.get(i, j) != 0.0) {
                    height = Board.NUM_ROWS - i;
                    return height;
                }
            }
        }
        return height;
    }

    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game,
                                 final GameCounter gameCounter)
    {
        // System.out.println("phaseIdx=" + gameCounter.getCurrentPhaseIdx() + "\tgameIdx=" + gameCounter.getCurrentGameIdx());
        // return this.getRandom().nextDouble() <= EXPLORATION_PROB;
        double noise = 0.05 * (random.nextDouble() - 0.5);
        double p = Math.exp(-0.0005 * gameCounter.getCurrentGameIdx()) * EXPLORATION_PROB;
        double probs = Math.max(0.2, p + noise);
        return probs > random.nextDouble();
    }

    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game)
    {
        List<Mino> possibleMoves = game.getFinalMinoPositions();
        Mino bestMino = null;
        int numClears = -1;

        if (possibleMoves.isEmpty()) {
            return null;
        }

        for (Mino mino : possibleMoves) {
            Matrix boardMatrix = null;

            try {
                boardMatrix = game.getGrayscaleImage(mino); // Generate the board state for the given move
            } catch (Exception e) {
                continue;
            }

            int newClears = calculateLinesCleared(boardMatrix);
            if (newClears > numClears) {
                bestMino = mino;
                numClears = newClears;
            }
        }

        if (bestMino == null) {
            double bestScore = Double.NEGATIVE_INFINITY;

            for (Mino mino : possibleMoves) {
                Matrix boardMatrix = null;

                try {
                    boardMatrix = game.getGrayscaleImage(mino);
                } catch (Exception e) {
                    continue;
                }

                // Use your existing feature calculation methods
                int linesCleared = calculateLinesCleared(boardMatrix);
                int holes = calculateHoles(boardMatrix);
                int bumps = calculateBumps(boardMatrix);
                int maxHeight = calculateMaxHeight(boardMatrix);
                double flatness = calculateFlatness(heights);
                double unevenness = calculateUnevenness(boardMatrix);

                double hval = 0.0;
                double fval = 0.0;

                if (maxHeight > 6) {
                    hval = -3.0;
                }

                if (flatness > 0) {
                    fval = -2.0;
                }
                
                // Calculate a weighted score for each move
                double score = (10.0 * linesCleared) + (fval * flatness) - (5.0 * holes) - (2.0 * bumps) - (2.0 * unevenness) + (hval * maxHeight);

                // Track the best scoring move
                if (score > bestScore) {
                    bestMino = mino;
                    bestScore = score;
                }
            }

            // Fallback: select a random move if no valid best move is found
            if (bestMino == null) {
                int randomIndex = this.getRandom().nextInt(possibleMoves.size());
                bestMino = possibleMoves.get(randomIndex);
            }
        }
        return bestMino;
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
    public double getReward(final GameView game)
    {
        double reward = 0.0;
        int score = game.getScoreThisTurn();
        // System.out.println(score);

        reward += 0.7 * linesCleared;
        reward += 0.5 * flatness;
        reward -= 0.2 * holes;
        reward -= 0.5 * bumps;
        reward -= 0.3 * maxHeight;
        reward -= 0.5 * unevenness;
        // System.out.println(reward);

        return reward;
    }
}