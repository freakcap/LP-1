import java.util.ArrayList;

public class Board {

    private final int[][] board; // n*n*4 bytes
    private int blankR; // 4bytes
    private int blankC; // 4 bytes
    private final int dim; // 4 bytes

    // create a board from an n-by-n array of tiles,
    // where tiles[row][col] = tile at (row, col)
    public Board(int[][] tiles) {
        if (tiles == null) {
            throw new NullPointerException();
        }
        this.dim = tiles.length;
        this.board = new int[this.dim][this.dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                this.board[i][j] = tiles[i][j];
                if (tiles[i][j] == 0) {
                    blankR = i;
                    blankC = j;
                }
            }
        }
    }

    // string representation of this board
    public String toString() {
        StringBuilder s = new StringBuilder();
        s.append(dim).append("\n");
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                s.append(String.format("%2d ", this.board[i][j]));
            }
            s.append("\n");
        }
        return s.toString();
    }

    // board dimension n
    public int dimension() {
        return this.dim;
    }

    // number of tiles out of place
    public int hamming() {

        int count = 0;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if (this.board[i][j] != 0) {
                    if (getManhattenDist(i, j) != 0)
                        count++;
                }
            }
        }
        return count;
    }

    private int getManhattenDist(int i, int j) {
        int expRow = (this.board[i][j] % this.dim == 0) ?
                (this.board[i][j] / this.dim - 1) :
                (this.board[i][j] / this.dim);
        int expCol = (this.board[i][j] % this.dim == 0) ?
                (this.dim - 1) :
                ((this.board[i][j] % this.dim) - 1);
        return Math.abs(expRow - i) + Math.abs(expCol - j);
    }

    // sum of Manhattan distances  Board(copy);between tiles and goal
    public int manhattan() {
        int count = 0;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if (this.board[i][j] != 0) {
                    count += getManhattenDist(i, j);
                }
            }
        }
        return count;
    }

    // is this board the goal board?
    public boolean isGoal() {
        // System.out.println("Checking goal");
        // System.out.println(board)
        for (int i = 0; i < this.dim; i++) {
            for (int j = 0; j < this.dim; j++) {
                if (i == this.dim - 1 && j == this.dim - 1) {
                    if (this.board[i][j] != 0)
                        return false;
                } else {
                    if (this.board[i][j] != (i * this.dim + j + 1))
                        return false;
                }
            }
        }
        return true;
    }

    // does this board equal y?
    public boolean equals(Object y) {
        if (y == this) return true;
        if (y == null) return false;
        if (y.getClass() != this.getClass()) return false;
        Board that = (Board) y;
        if (that.dim != this.dim) return false;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if (that.board[i][j] != this.board[i][j])
                    return false;
            }
        }
        return true;
    }

    // all neighboring boards
    public Iterable<Board> neighbors() {
        ArrayList<Board> sB = new ArrayList<Board>();
        // check if there is top neighbour
        if (blankR - 1 >= 0) {
            sB.add(swap(blankR - 1, blankC, blankR, blankC));
        }
        // right neighbour
        if (blankC + 1 < this.dim) {
            sB.add(swap(blankR, blankC + 1, blankR, blankC));

        }
        // bottom neighbour
        if (blankR + 1 < this.dim) {
            sB.add(swap(blankR + 1, blankC, blankR, blankC));

        }
        // left neighbour
        if (blankC - 1 >= 0) {
            sB.add(swap(blankR, blankC - 1, blankR, blankC));
        }
        return sB;

    }

    private Board swap(int r1, int c1, int r2, int c2) {
        assert r1 >= 0 && r1 < dim;
        assert r2 >= 0 && r2 < dim;
        assert c1 >= 0 && c1 < dim;
        assert c2 >= 0 && c2 < dim;
        int temp;
        temp = this.board[r1][c1];
        this.board[r1][c1] = this.board[r2][c2];
        this.board[r2][c2] = temp;
        Board copy = new Board(this.board);
        temp = this.board[r1][c1];
        this.board[r1][c1] = this.board[r2][c2];
        this.board[r2][c2] = temp;
        return copy;
    }

    public Board twin() {
        int[][] copy = new int[this.dim][this.dim];
        for (int i = 0; i < this.dim; i++)
            for (int j = 0; j < this.dim; j++)
                copy[i][j] = this.board[i][j];
        if (this.blankR != 0) {
            int temp = copy[0][0];
            copy[0][0] = copy[0][1];
            copy[0][1] = temp;
        } else {
            int temp = copy[1][0];
            copy[1][0] = copy[1][1];
            copy[1][1] = temp;
        }
        return new Board(copy);
    }


    // unit testing (not graded)
    public static void main(String[] args) {
//        In in = new In(args[0]);
//        int n = in.readInt();
//        int[][] tiles = new int[n][n];
//        for (int i = 0; i < n; i++)
//            for (int j = 0; j < n; j++)
//                tiles[i][j] = in.readInt();
//        Board initial = new Board(tiles);
//        Board t = initial.twin();
//        System.out.println(initial);
//        System.out.println(t);
    }
}