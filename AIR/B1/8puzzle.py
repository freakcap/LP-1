import queue as q
import copy

class puzzle:

    def __init__(self):
        self.size=0
        self.num_pos={}
        self.visited=set()
        self.initial_state=[]
        self.final_state=[]
        self.trace_game={}
        self.st=q.PriorityQueue()
        print("Enter the Size of the Puzzle:")
        self.size=int(input())
        print("Enter Initial State:")
        for i in range(self.size):
            li=list(map(int,input().split(" ")))
            self.initial_state.append(li)
        print("Enter Goal State:")
        for i in range(self.size):
            lf=list(map(int,input().split(" ")))
            self.final_state.append(lf)
        for i in range(self.size):		# store the positions of numbers in final states
            for j in range(self.size):
                self.num_pos[self.final_state[i][j]] = (i,j)

    def move_validate(self,move):
        if move[0]<0 or move[1]>=self.size or move[0]>=self.size or move[1]<0:
            return False
        return True
        
    def zero_pos(self,board):
        for i in range(self.size):
            for j in range(self.size):
                if(board[i][j]==0):
                    return (i,j)
                    
    def move_generate(self,board):
        moves=list()
        position=self.zero_pos(board)
        x,y=position[0],position[1]
        if self.move_validate((x+1,y)):
            moves.append((x+1,y))
        if self.move_validate((x-1,y)):
            moves.append((x-1,y))
        if self.move_validate((x,y+1)):
            moves.append((x,y+1))
        if self.move_validate((x,y-1)):
            moves.append((x,y-1))
        return moves
        
    def heuristic(self,board):
        cost=0
        for i in range(self.size):
            for j in range(self.size):
                position=self.num_pos[board[i][j]]
                cost=cost+(abs(i-position[0])+abs(j-position[1]))
        return cost
        
    def convert_list_to_tuple(self,board):
        list_to_tuple=tuple(tuple(i) for i in board)
        return list_to_tuple
        
    def convert_tuple_to_list(self,board):
        board_config=list()
        for i in board:
            board_config.append(list(i))
        return board_config
        
    def print_grid(self,board):
        for i in board:
            for j in i:
                print(j,end=" ")
            print()
            
    def print_move_sequence(self):
        move_sequence = []
        cur = self.convert_list_to_tuple(self.final_state)
        while cur!=self.trace_game[cur]:
            move_sequence.append(cur)
            cur = self.trace_game[cur]
        move_sequence.append(self.initial_state)
        move_sequence.reverse()
        for i in move_sequence:
            self.print_grid(i)
            print()
            
    def play(self):
        iterations=0
        self.st.put(((0,0),self.convert_list_to_tuple(self.initial_state)))
        self.trace_game[self.convert_list_to_tuple(self.initial_state)]=self.convert_list_to_tuple(self.initial_state)
        while self.st.qsize()>0:
            current_state=self.st.get()
            board=self.convert_tuple_to_list(current_state[1])
            if self.convert_list_to_tuple(board) in self.visited:
                continue
            self.visited.add(self.convert_list_to_tuple(board))
            if board==self.final_state:
                print("The number of moves taken to reach goal state:"+str(current_state[0][1]))
                return
            moves=self.move_generate(board)
            pos=self.zero_pos(board)
            for move in moves:
                duplicate=copy.deepcopy(board)
                temp=duplicate[pos[0]][pos[1]]
                duplicate[pos[0]][pos[1]]=duplicate[move[0]][move[1]]
                duplicate[move[0]][move[1]]=temp
                if self.convert_list_to_tuple(duplicate) not in self.visited:
                    step=current_state[0][1]
                    self.trace_game[self.convert_list_to_tuple(duplicate)]=self.convert_list_to_tuple(board)
                    self.st.put(((step+1+self.heuristic(duplicate),step+1),self.convert_list_to_tuple(duplicate)))
            iterations+=1
            if iterations>10**6:
                print("Probably not solvable")
                return

game=puzzle()
game.play()
game.print_move_sequence()
