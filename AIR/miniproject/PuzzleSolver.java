/* *****************************************************************************
 *  Name:
 *  Date:
 *  Description:
 **************************************************************************** */


import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class PuzzleSolver {
    private boolean solvable;
    private SearchNode solution = null;

    private class SearchNode implements Comparable<SearchNode> {

        private  Board board;
        private  SearchNode parent;
        private final int moves;
        private final int h;

        public SearchNode(Board board, SearchNode parent, int moves) {
            this.board = board;
            this.parent = parent;
            this.moves = moves;
            this.h = board.hamming();
        }

        public int compareTo(SearchNode searchNode) {
            return Integer
                    .compare(this.h + this.moves, searchNode.h + searchNode.moves);
        }
    }

    public PuzzleSolver(Board initial) {
        if (initial == null)
            throw new IllegalArgumentException();
        // System.out.println(initial);
        // find a solution to the initial board (using the A* algorithm)



        SearchNode startState = new SearchNode(initial, null, 0);

        //If startState is Goal State 
        if(startState.board.isGoal()){
            this.solvable = true;
            this.solution = startState;
            return;
        }
        SearchNode currentState = startState;
        
        //else
        // pqOrig.add(new SearchNode(initial, null, 0));
        //
        // System.out.println(pqOrig.isEmpty());
        // System.out.println(pqTwin.isEmpty());

        while (true) {
            //Poll the lowest heuristic state.
            SearchNode nextState=null;
            Integer nextH = Integer.MAX_VALUE;

            for (Board b : currentState.board.neighbors()) {
                if (currentState.parent != null && b.equals(currentState.parent.board))
                    continue;
                if(b.hamming() < nextH){
                    if(nextState==null)
                        nextState= new SearchNode(b, currentState, currentState.moves+1);
                    else
                        nextState.board=b;
                    nextH = nextState.board.hamming();
                    
                }
               
                // pqOrig.add(new SearchNode(b, minBoardOrig, minBoardOrig.moves + 1));
            }

            //if no neighbour is better than current neighbour exit
            if(nextH > currentState.board.hamming()){
                this.solvable = true;
                this.solution = currentState;
                return;
            }


            nextState.parent = currentState;
            currentState = nextState;
            // SearchNode minBoardOrig = pqOrig.poll();
            // if (minBoardOrig.board.isGoal()) {
            //     // System.out.println("Found Goal");
            //     this.solvable = true;
            //     this.solution = minBoardOrig;
            //     return;
            // }
            // for (Board b : minBoardOrig.board.neighbors()) {
            //     if (minBoardOrig.parent != null && b.equals(minBoardOrig.parent.board))
            //         continue;
            //     pqOrig.add(new SearchNode(b, minBoardOrig, minBoardOrig.moves + 1));
            // }
            // for (Board b : minBoardTwin.board.neighbors()) {
            //     if (minBoardTwin.parent != null && b.equals(minBoardTwin.parent.board))
            //         continue;
            //     pqTwin.add(new SearchNode(b, minBoardTwin, minBoardTwin.moves + 1));
            // }
        }
    }

    // is the initial board solvable? (see below)
    public boolean isSolvable() {
        return this.solvable;
    }

    // min number of moves to solve initial board; -1 if unsolvable
    public int moves() {
        return (this.solution != null) ? this.solution.moves : -1;
    }

    // sequence of boards in a shortest solution; null if unsolvable
    public Iterable<Board> solution() {

        if (this.solution == null)
            return null;
        ArrayList<Board> path = new ArrayList<>();
        SearchNode temp = this.solution;
        while (temp != null) {
            path.add(temp.board);
            temp = temp.parent;
        }
        Collections.reverse(path);
        return path;
    }

    // test client (see below)
    public static void main(String[] args) throws FileNotFoundException {

        // create initial board from file
        Scanner in = new Scanner(new FileInputStream(args[0]));
        int[][] tiles = new int[0][];
        while (in.hasNext()) {
            int n = in.nextInt();
            tiles = new int[n][n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    tiles[i][j] = in.nextInt();
        }
        Board initial = new Board(tiles);

        // solve the puzzle
        PuzzleSolver solver = new PuzzleSolver(initial);

        // print solution to standard output
        if (!solver.isSolvable())
            System.out.println("No solution possible");
        else {
            System.out.println("Minimum number of moves = " + solver.moves());
            for (Board board : solver.solution()){
                System.out.println(board);
                System.out.println(board.hamming());
            }
        }
    }
}
